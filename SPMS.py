
# -*- coding: utf-8 -*-
"""
Created on Sun May 10 15:30:18 2020
Modified on Fri Jan 19 15:20:10 2024

@author: Licheng Xu and Shuoqing Zhang
"""

from rdkit import Chem
import numpy as np
from ase.io import read as ase_read
from copy import deepcopy
from rot import tript2fix_orit
precision = 8
class SPMS():
    def __init__(self,sdf_file=None,xyz_file=None,key_atom_num=None,sphere_radius=None,desc_n=40,desc_m=40,
                 orientation_standard=True,first_point_index_list=None,second_point_index_list=None,third_point_index_list=None):
        '''
        
        Parameters
        ----------
        sdf_file : string
            path of .sdf file.
        xyz_file : string
            path of .xyz file.
        key_atom_num: list
            List of key atomic indices for constraining molecular orientation.
        sphere_radius: float
            Radius of the spherical surface enclosing the molecule.
        desc_n: int
            Descriptor's latitudinal resolution
        desc_m: int
            Descriptor's longitudinal resolution
        orientation_standard: True, False or 'Customized'
            if the molecular orientation is standardized
        
        Returns
        -------
        None.

        '''
        assert sdf_file != None or xyz_file != None, "You must input one of 'sdf_file' and 'xyz_file'"
        self.precision = 8
        self.sdf_file = sdf_file
        self.xyz_file = xyz_file
        self.sphere_radius = sphere_radius
        if key_atom_num != None:
            key_atom_num = list(np.array(key_atom_num,dtype=np.int)-1)
            self.key_atom_num = key_atom_num
        else:
            self.key_atom_num = []
        self.desc_n = desc_n
        self.desc_m = desc_m
        self.orientation_standard = orientation_standard
        self.first_point_index_list = first_point_index_list
        self.second_point_index_list = second_point_index_list
        self.third_point_index_list = third_point_index_list

        rdkit_period_table = Chem.GetPeriodicTable()
        if self.sdf_file is not None:
            mol = Chem.MolFromMolFile(sdf_file,removeHs=False,sanitize=False)
            conformer = mol.GetConformer()
            positions = conformer.GetPositions()
            atoms = mol.GetAtoms()
            atom_types = [atom.GetAtomicNum() for atom in atoms]
            atom_symbols = [rdkit_period_table.GetElementSymbol(item) for item in atom_types]
            atom_weights = [atom.GetMass() for atom in atoms]
        elif self.xyz_file is not None:
            ase_atoms = ase_read(self.xyz_file,format='xyz')
            positions = ase_atoms.get_positions()
            atom_symbols = list(ase_atoms.symbols)
            atom_types = [rdkit_period_table.GetAtomicNumber(sym) for sym in atom_symbols]
            atom_weights = [rdkit_period_table.GetAtomicWeight(sym) for sym in atom_symbols]
            
        
        atom_weights = np.array([atom_weights,atom_weights,atom_weights]).T
        weighted_pos = positions*atom_weights
        weight_center = np.round(weighted_pos.sum(axis=0)/atom_weights.sum(axis=0)[0],decimals=self.precision)
        radius = np.array([rdkit_period_table.GetRvdw(item) for item in atom_types]) # van der Waals radius
        volume = 4/3*np.pi*pow(radius,3)
        self.positions = positions
        self.weight_center = weight_center
        self.radius = radius
        self.volume = volume
        self.atom_types = atom_types
        self.atom_symbols = atom_symbols
        self.rdkit_period_table = rdkit_period_table
        self.atom_weight = atom_weights
    def _Standarlize_Geomertry_Input(self,origin_positions):
        
        if self.key_atom_num == []:
            key_atom_position = deepcopy(self.weight_center)
            key_atom_position = key_atom_position.reshape(1,3)
            distmat_from_key_atom = np.sqrt(np.sum((origin_positions - key_atom_position)**2,axis=1))
            farest_atom_index = np.argmax(distmat_from_key_atom)
            distmat_from_key_atom[np.argmin(distmat_from_key_atom)] = np.max(distmat_from_key_atom)
            nearest_atom_index = np.argmin(distmat_from_key_atom)
            second_key_atom_index = nearest_atom_index
            third_key_atom_index = farest_atom_index
            second_atom_position = deepcopy(origin_positions[second_key_atom_index])
            second_atom_position = second_atom_position.reshape(1,3)
            third_atom_position = deepcopy(origin_positions[third_key_atom_index])
            third_atom_position = third_atom_position.reshape(1,3)
            append_positions = np.concatenate([origin_positions,key_atom_position,second_atom_position,third_atom_position])
        else:
            key_atom_num = self.key_atom_num
            if len(key_atom_num) == 1:
                key_atom_position = deepcopy(origin_positions[key_atom_num[0]])
                key_atom_position = key_atom_position.reshape(1,3)
                distmat_from_key_atom = np.sqrt(np.sum((origin_positions - key_atom_position)**2,axis=1))
                farest_atom_index = np.argmax(distmat_from_key_atom)
                distmat_from_key_atom[np.argmin(distmat_from_key_atom)] = np.max(distmat_from_key_atom)
                nearest_atom_index = np.argmin(distmat_from_key_atom)
                second_key_atom_index = nearest_atom_index
                third_key_atom_index = farest_atom_index
                second_atom_position = deepcopy(origin_positions[second_key_atom_index])
                second_atom_position = second_atom_position.reshape(1,3)
                third_atom_position = deepcopy(origin_positions[third_key_atom_index])
                third_atom_position = third_atom_position.reshape(1,3)
                append_positions = np.concatenate([origin_positions,key_atom_position,second_atom_position,third_atom_position])

            elif len(key_atom_num) >= 2:
                key_atom_position = origin_positions[key_atom_num].mean(axis=0)
                key_atom_position = key_atom_position.reshape(1,3)
                distmat_from_key_atom = np.sqrt(np.sum((origin_positions - key_atom_position)**2,axis=1))
                farest_atom_index = np.argmax(distmat_from_key_atom)
                distmat_from_key_atom[np.argmin(distmat_from_key_atom)] = np.max(distmat_from_key_atom)
                nearest_atom_index = np.argmin(distmat_from_key_atom)
                second_key_atom_index = nearest_atom_index
                third_key_atom_index = farest_atom_index
                second_atom_position = deepcopy(origin_positions[second_key_atom_index])
                second_atom_position = second_atom_position.reshape(1,3)
                third_atom_position = deepcopy(origin_positions[third_key_atom_index])
                third_atom_position = third_atom_position.reshape(1,3)
                append_positions = np.concatenate([origin_positions,key_atom_position,second_atom_position,third_atom_position])

        new_positions = tript2fix_orit(append_positions,[len(origin_positions)],[len(origin_positions)+1],[len(origin_positions)+2],axis='z',plane='yz')
        std_positions = new_positions[:-3]
        std_3points = new_positions[-3:]
        return std_positions,std_3points        
        
    def _Customized_Coord_Standard(self,positions,first_point_index_list,second_point_index_list,third_point_index_list):
        
        first_point_index_list = [item-1 for item in first_point_index_list]
        second_point_index_list = [item-1 for item in second_point_index_list]
        third_point_index_list = [item-1 for item in third_point_index_list]
        new_positions = tript2fix_orit(positions,first_point_index_list,second_point_index_list,third_point_index_list,axis='z',plane='yz')
        return new_positions
    
    
    
    def _Standarlize_Geomertry(self):
        if self.orientation_standard == True:
            new_positions,new_3points = self._Standarlize_Geomertry_Input(self.positions)
            if self.key_atom_num != None:
                bias_move = np.array([0.000001,0.000001,0.000001])
                new_positions += bias_move
                new_3points += bias_move
            new_geometric_center,new_weight_center,new_weight_center_2 = new_3points[0],new_3points[1],new_3points[2]
            self.new_positions = new_positions
            self.new_geometric_center,self.new_weight_center,self.new_weight_center_2 = new_geometric_center,new_weight_center,new_weight_center_2
        
        elif self.orientation_standard == False:
            new_positions = self.positions
            self.new_positions = self.positions
        
        elif self.orientation_standard == "Customized":
            new_positions = self._Customized_Coord_Standard(self.positions,self.first_point_index_list,self.second_point_index_list,self.third_point_index_list)
            self.new_positions = new_positions
        distances = np.sqrt(np.sum(new_positions**2,axis=1))
        self.distances = distances
        distances_plus_radius = distances + self.radius
        sphere_radius = np.ceil(distances_plus_radius.max())
        if self.sphere_radius == None:
            self.sphere_radius = sphere_radius
        
    def _polar2xyz(self,r,theta,phi):
        x = r*np.sin(theta)*np.cos(phi)
        y = r*np.sin(theta)*np.sin(phi)
        z = r*np.cos(theta)
        return np.array([x,y,z])
    def _xyz2polar(self,x,y,z):
        # theta 0-pi
        # phi 0-2pi
        r = np.sqrt(x**2+y**2+z**2)
        theta = np.arcsin(np.sqrt(x**2+y**2)/r)
        phi = np.arctan(y/x)
        if z < 0:
            theta = np.pi - theta
        if x < 0 and y > 0:
            phi = np.pi + phi
        elif x < 0 and y < 0:
            phi = np.pi + phi
        elif x > 0 and y < 0:
            phi = 2*np.pi + phi
        return np.array([r,theta,phi])
    def Writegjf(self,file_path):
        try:
            new_positions = self.new_positions
        except:
            self._Standarlize_Geomertry()
            new_positions = self.new_positions
        atom_types = self.atom_types
        coord_string = ''
        for at_ty,pos in zip(atom_types,new_positions):
            coord_string += '%10d %15f %15f %15f \n'%(at_ty,pos[0],pos[1],pos[2])
        string = '#p\n\nT\n\n0 1\n' + coord_string + '\n'
        with open(file_path,'w') as fw:
            fw.writelines(string)
            
            
    def Writexyz(self,file_path):
        try:
            new_positions = self.new_positions
        except:
            self._Standarlize_Geomertry()
            new_positions = self.new_positions
        atom_symbols = self.atom_symbols
        
        atom_num = len(atom_symbols)
        coord_string = '%d\ntitle\n'%atom_num
        for at_sy,pos in zip(atom_symbols,new_positions):
            coord_string += '%10s %15f %15f %15f \n'%(at_sy,pos[0],pos[1],pos[2])
        with open(file_path,'w') as fw:
            fw.writelines(coord_string)
    def GetSphereDescriptors(self):
        self._Standarlize_Geomertry()
        new_positions = self.new_positions
        
        radius = self.radius
        sphere_radius = self.sphere_radius

        N = self.desc_n
        M = self.desc_m
        delta_theta = 1/N * np.pi
        delta_phi = 1/M * np.pi
        theta_screenning = np.array([item*delta_theta for item in range(1,N+1)])
        self.theta_screenning = theta_screenning
        phi_screenning = np.array([item*delta_phi for item in range(1,M*2+1)])
        self.phi_screenning = phi_screenning
        PHI, THETA = np.meshgrid(phi_screenning, theta_screenning)

        x = sphere_radius*np.sin(THETA)*np.cos(PHI)
        y = sphere_radius*np.sin(THETA)*np.sin(PHI)
        z = sphere_radius*np.cos(THETA)
        mesh_xyz = np.array([[x[i][j],y[i][j],z[i][j]] for i in range(theta_screenning.shape[0]) for j in range(phi_screenning.shape[0])])
        self.mesh_xyz = mesh_xyz
        psi = np.linalg.norm(new_positions,axis=1)
        atom_vec = deepcopy(new_positions)
        self.psi = psi
        all_cross = []
        for j in range(atom_vec.shape[0]): #######################
            all_cross.append(np.cross(atom_vec[j].reshape(-1,3),mesh_xyz,axis=1)) #############
        all_cross = np.array(all_cross)
        all_cross = all_cross.transpose(1,0,2)
        self.all_cross = all_cross
        mesh_xyz_h = np.linalg.norm(all_cross,axis=2)/sphere_radius

        dot = np.dot(mesh_xyz,atom_vec.T)
        atom_vec_norm = np.linalg.norm(atom_vec,axis=1).reshape(-1,1)
        mesh_xyz_norm = np.linalg.norm(mesh_xyz,axis=1).reshape(-1,1)
        self.mesh_xyz_norm = mesh_xyz_norm
        self.atom_vec_norm = atom_vec_norm
        
        orthogonal_mesh = dot/np.dot(mesh_xyz_norm,atom_vec_norm.T)
        
        self.mesh_xyz_h = mesh_xyz_h
        
        self.orthogonal_mesh = orthogonal_mesh
        
        #cross_det
        cross_det = mesh_xyz_h <= radius
        #orthogonal_det
        orthogonal_det = np.arccos(orthogonal_mesh) <= np.pi*0.5
        double_correct = np.array([orthogonal_det,cross_det]).all(axis=0)
        double_correct_index = np.array(np.where(double_correct==True)).T
        self.double_correct_index = double_correct_index
        d_1 = np.zeros(mesh_xyz_h.shape)
        d_2 = np.zeros(mesh_xyz_h.shape)
        for item in double_correct_index:
            
            d_1[item[0]][item[1]] = max( (psi[item[1]]**2 - mesh_xyz_h[item[0]][item[1]]**2) ,0)**0.5
            d_2[item[0]][item[1]]=(radius[item[1]]**2 - mesh_xyz_h[item[0]][item[1]]**2)**0.5
        self.d_1 = d_1
        self.d_2 = d_2
        
        sphere_descriptors = sphere_radius - d_1 - d_2
        sphere_descriptors_compact = sphere_descriptors.min(1)
        sphere_descriptors_reshaped = sphere_descriptors_compact.reshape(PHI.shape)
        sphere_descriptors_reshaped = sphere_descriptors_reshaped.round(self.precision)
        
        if len(self.key_atom_num) == 1:
            sphere_descriptors_init = np.zeros((theta_screenning.shape[0],phi_screenning.shape[0])) + sphere_radius - self.radius[self.key_atom_num[0]]
            sphere_descriptors_final = np.min(np.concatenate([sphere_descriptors_reshaped.reshape(theta_screenning.shape[0],phi_screenning.shape[0],1),sphere_descriptors_init.reshape(theta_screenning.shape[0],phi_screenning.shape[0],1)],axis=2),axis=2)
        else:
            sphere_descriptors_final = sphere_descriptors_reshaped
        
        self.PHI = PHI
        self.THETA = THETA
        self.sphere_descriptors = sphere_descriptors_final
        return self.sphere_descriptors
    def GetQuaterDescriptors(self):
        try:
            sphere_descriptors = self.sphere_descriptors
        except:
            sphere_descriptors = self.GetSphereDescriptors()

        self.left_top_desc = sphere_descriptors[:self.desc_n//2,:self.desc_m]
        self.right_top_desc = sphere_descriptors[:self.desc_n//2,self.desc_m:]
        self.left_bottom_desc = sphere_descriptors[self.desc_n//2:,:self.desc_m]
        self.right_bottom_desc = sphere_descriptors[self.desc_n//2:,self.desc_m:]

        self.left_top_desc_sum = self.left_top_desc.sum()
        self.right_top_desc_sum = self.right_top_desc.sum()
        self.left_bottom_desc_sum = self.left_bottom_desc.sum()
        self.right_bottom_desc_sum = self.right_bottom_desc.sum()

        _sum = sphere_descriptors.sum()

        self.left_top_desc_partial = self.left_top_desc_sum/_sum
        self.right_top_desc_partial = self.right_top_desc_sum/_sum
        self.left_bottom_desc_partial = self.left_bottom_desc_sum/_sum
        self.right_bottom_desc_partial = self.right_bottom_desc_sum/_sum

        return self.left_top_desc_partial,self.right_top_desc_partial,\
               self.left_bottom_desc_partial,self.right_bottom_desc_partial
