from setuptools import find_packages, setup

setup(name="psd_gnn",
      version="0.0.1",
      author="PoSeiDon project",
      summary="TBD",
      license="MIT",
      author_email="jinh@anl.gov",
      packages=find_packages(exclude=["tests", "results", "log", "psd.egg-info"]))
