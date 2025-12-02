from setuptools import setup, find_packages

setup(
    name="KTRtracker",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        'seaborn',
        'tifffile',
        'scikit-image',
        'cellpose',
        'scipy'
    ],
    extras_require={
        'napari': ['napari'],
        'dev': ['pytest']
    },
    author="Yuhei Goto",
    author_email="goto.yuhei.4c@kyoto-u.ac.jp",
    description="A comprehensive image analysis and tracking package for KTR",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yugo8849/KTRtracker",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)