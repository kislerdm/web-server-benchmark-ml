from setuptools import setup, find_packages

setup(
    name='web_server_benchmark_mnist',
    version='1.0',
    description='Common layer of the web-server-benchmark-mnist',
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    author='Dmitry Kisler',
    author_email='admin@dkisler.com',
    license='MIT',
    packages=find_packages(),
    install_requires=[
        "tensorflow==2.0.0b1",
        "opencv-python==4.1.0.25",
        "numpy>=1.16.3",
    ])
