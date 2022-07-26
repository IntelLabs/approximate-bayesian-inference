from setuptools import setup

setup(
    name='pyViewer',
    version='0.1.0',
    author='Javier Felip Leon',
    author_email='javier.felip.leon@gmail.com',
    packages=['pyViewer'],
    data_files=[('pyViewer/fonts', ['pyViewer/fonts/FiraCode-Medium.ttf']),
                ('pyViewer/models/camera_gizmo', ['pyViewer/models/camera_gizmo/camera_gizmo.obj']),
                ('pyViewer/models/camera_gizmo', ['pyViewer/models/camera_gizmo/camera_gizmo.mtl']),
                ('pyViewer/models/floor40x40', ['pyViewer/models/floor40x40/floor40x40.obj']),
                ('pyViewer/models/floor40x40', ['pyViewer/models/floor40x40/floor40x40.mtl']),
                ('pyViewer/models/floor40x40', ['pyViewer/models/floor40x40/grid-1m0.1cm.png']),
                ('pyViewer/models/reference_frame', ['pyViewer/models/reference_frame/reference_frame.obj']),
                ('pyViewer/models/reference_frame', ['pyViewer/models/reference_frame/reference_frame.mtl'])],
    include_package_data=True,
    scripts=[],
    url='',
    license='LICENSE',
    description='Simple modern openGL viewer',
    long_description=open('README.md').read(),
    install_requires=[
        "numpy",
        "pywavefront",
        "moderngl",
        "pyopengl",
        "pillow",
        "pyglfw",
        "transforms3d"
    ],
)