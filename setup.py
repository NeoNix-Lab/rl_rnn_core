from setuptools import setup, find_packages

setup(
    name="rl_rnn_core",
    version="0.1.0",
    description="Moduli RL per trading con Gym, buffer, modelli e DB",
    author="Neo-Nix Lab",
    packages=find_packages(),
    install_requires=[
        "tensorflow>=2.16.1",
        "keras>=3.2.1",
        "gym>=0.26.2",
        "numpy>=1.26.4",
        "pandas>=1.5.3",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
)
