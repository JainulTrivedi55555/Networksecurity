from setuptools import setup, find_packages
from typing import List

def get_requirements() -> List[str]:
    """
    This function will return the list of the requirements
    """
    requirement_lst: List[str] = []
    try:
        with open('requirements.txt') as file:
            ## Read lines from the file
            lines = file.readlines()
            ## Process each line
            for line in lines:
                requirement = line.strip()
                ## ignore empty lines and -e.
                if requirement and requirement != '-e .':
                    requirement_lst.append(requirement)
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
    
    return requirement_lst

setup(
    name='NetworkSecurity',
    version='0.0.1',
    author='Jainul Trivedi',
    author_email="jainultrivedi55555@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements()
)
