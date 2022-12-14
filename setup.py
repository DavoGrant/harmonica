from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext


ext_modules = [
    Pybind11Extension(
        "harmonica/bindings",
        [
             'harmonica/orbit/kepler.cpp',
             'harmonica/orbit/trajectories.cpp',
             'harmonica/orbit/gradients.cpp',
             'harmonica/light_curve/fluxes.cpp',
             'harmonica/light_curve/gradients.cpp',
             'harmonica/bindings.cpp'
         ],
        include_dirs=["vendor/eigen", "vendor/pybind11"],
        language="c++",
        extra_compile_args=["-O2", "-ffast-math"]
    ),
]

setup(
    name="planet-harmonica",
    version="0.1.0",
    author="David Grant",
    author_email="david.grant@bristol.ac.uk",
    url="https://github.com/DavoGrant/harmonica",
    license="MIT",
    packages=["harmonica", "harmonica.jax"],
    description="Light curves for exoplanet transmission mapping.",
    long_description="Light curves for exoplanet transmission mapping.",
    python_requires=">=3.6",
    install_requires=["numpy", "jax", "jaxlib"],
    cmdclass={"build_ext": build_ext},
    ext_modules=ext_modules,
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Topic :: Software Development :: Libraries :: Python Modules"
    ],
)
