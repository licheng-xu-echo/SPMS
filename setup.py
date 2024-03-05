from setuptools import setup, find_packages

setup(
    name='spms',  
    version='0.1',  
    packages=find_packages("."),  
    description='Package for generating SPMS descriptor',
    long_description=open('README.md').read(),
    author='Li-Cheng Xu', 
    author_email='licheng_xu@zju.edu.cn',  
    url='https://github.com/licheng-xu-echo/SPMS',  
    install_requires=[  
        'numpy >= 1.17.4',
        'ase >= 3.19.1',
    ],
    license="MIT",
    python_requires=">=3.6",
)