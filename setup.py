import setuptools
  
with open("README.md", "r") as fh:
    description = fh.read()
  
setuptools.setup(
    name="SPO_Plus_Training",
    version="0.0.1",
    author="Krishna Kalathur",
    author_email="krishna.kalathur@gmail.com",
    packages=["SPO_Plus_Training"],
    description="A sample test package",
    long_description=description,
    long_description_content_type="text/markdown",
    url="https://github.com/1362136/SPO-Plus-Training-Package",
    license='MIT',
    python_requires='>=3.8',
    install_requires=[]
)