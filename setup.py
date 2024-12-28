from setuptools import setup, find_packages

# Replace these values with details specific to your project
PACKAGE_NAME = "curriculum_compass"
VERSION = "0.0.1"
AUTHOR="Kishore Sampath, Pratheesh"
AUTHOR_EMAIL = "skishore2602.dev@gmail.com, author2.email@example.com"
DESCRIPTION = "Curriculum Compass is a conversational AI chatbot designed to assist Northeastern University graduate students with course registration."
URL = "https://github.com/Kishore-Pratheesh-AI-Projects/course-registration-chatbot"
REQUIRES_PYTHON = ">=3.7"

# Read dependencies from requirements.txt
with open("requirements.txt") as f:
    INSTALL_REQUIRES = f.read().splitlines()

setup(
    name=PACKAGE_NAME,
    version=VERSION,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    description=DESCRIPTION,
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url=URL,
    packages=find_packages(exclude=["chromadb"]),
    python_requires=REQUIRES_PYTHON,
    install_requires=INSTALL_REQUIRES,
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
)
