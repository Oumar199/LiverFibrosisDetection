from setuptools import setup, find_packages

setup(
    name="liverfibrosisdetection",
    version="0.0.1",
    author="Cheikh Yakhoub Maas, Mamadou Bousso, Oumar Kane, Methou Sanghe, Aby Diallo",
    packages=find_packages(),
    author_email="cyakhoub.maas@univ-thies.sn ",
    description="Contains modules to detect liver fibrosis level and analysis the results from both heatmap of the liver fibrosis and clinical data using advanced AI models.",
    install_requires=[
        "torch==2.10.0+cu128",
        "numpy==2.0.2",
        "pandas==2.2.2",
        "matplotlib==3.10.0",
        "pillow==11.3.0",
        "opencv-python==4.13.0",
        "json==2.0.9",
        "scikit-learn==1.6.1",
        "torchvision==0.25.0+cu128",
        "transformers==5.0.0",
        "IPython==7.34.0"
    ],
)
