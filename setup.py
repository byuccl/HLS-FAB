from setuptools import setup, find_packages

setup(
    name='hlseval',
    version='0.1',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "CppHeaderParser",
        "openai",
        "numpy",
        "requests",
        "torch",
        "llama_index", 
        "ml_dtypes",
        "scipy",
    ]
)