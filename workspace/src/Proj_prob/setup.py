from setuptools import find_packages, setup

package_name = 'Proj_prob'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='robot',
    maintainer_email='robot@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'no_posicao = Proj_prob.posicao:main',
            'no_ekf = Proj_prob.EKF:main',
            'mapeamento = Proj_prob.mapeamento:main',
            'lidar_to_grid_map = Proj_prob.lidar_to_grid_map:main',
        ],
    },
)
