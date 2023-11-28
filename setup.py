import setuptools

setuptools.setup(
    name='hpacker',
    version='0.1.0',
    author='Gian Marco Visani',
    author_email='gvisan01@.cs.washington.edu',
    description='Holographic Rotationally Equivariant Convolutional Neural Network for Protein Side-Chain Packing',
    long_description=open("README.md", "r").read(),
    long_description_content_type='text/markdown',
    url='https://github.com/gvisani/hpacker',
    python_requires='>=3.9',
    install_requires='',
    packages=setuptools.find_packages(),
)