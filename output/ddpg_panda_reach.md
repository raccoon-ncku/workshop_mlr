name: ws_mlr
channels:
  - defaults
  - conda-forge
dependencies:
  - python=3.10
  - pip
  - ipykernel
  - numpy
  - scipy
  - swig # gym dependency
  - moviepy # gym recording wrapper
  - pybullet
  - pip:
    - gymnasium==0.29.1
    - 'stable-baselines3[extra]==2.3.2'
    - -e external_modules/panda-gym