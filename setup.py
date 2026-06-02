from setuptools import setup, find_packages

setup(
    name="quadrotor-delivery",
    version="0.1.0",
    description="Multi-agent quadrotor delivery task assignment environment",
    author="Researcher",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.24.0",
        "gymnasium>=0.29.0",
    ],
    extras_require={
        "viz": ["matplotlib>=3.7.0", "imageio>=2.20.0", "imageio-ffmpeg>=0.4.0"],
        "rl": ["stable-baselines3>=2.1.0", "torch>=2.0.0"],
        "marl": ["ray[rllib]>=2.0.0"],
        "dev": ["pytest>=7.0.0"],
        "all": [
            "matplotlib>=3.7.0",
            "imageio>=2.20.0",
            "imageio-ffmpeg>=0.4.0",
            "stable-baselines3>=2.1.0",
            "torch>=2.0.0",
            "pytest>=7.0.0",
        ],
    },
)
