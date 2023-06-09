<p align="center">
<img
  src="./Data/diffrapy_logo.png"
  alt="Alt text"
  style="display: inline-block">
  </p>

## Summary

[Introduction to the Software](#introduction)

[Homogeneous model](#homogeneous-model)

[Heterogeneous model](#heterogeneous-model)

[Writing Reading SEGY files](#segy-files)

[Have a question or found a problem?](#have-a-question-or-found-a-problem)

## Introduction

We present DiffraPy, a Python based software for processing and imaging seismic diffractions. 

We show how to run the functions in the .ipynb examples, i.e., interactive Jupyter Notebooks.

Try creating your own!

We highly recommend to use Anaconda (https://www.anaconda.com/).

All main dependencies are listed below:
  
   ```
   python version = 3.8.8
   obspy = 1.3.0
   numpy = 1.19.5
   scipy = 1.8.0
   matplotlib = 3.5.1
   ```

## Homogeneous model

In the notebook ```Homogeneous_VelocityModel_Example.ipynb```, you can find the example of a toy model for a general understanding of the method.

## Heterogeneous model

In the notebook ```Heterogeneous_VelocityModel_Example.ipynb```, you can find the example of a complex synthetic model based on the SEG/EAGE salt model. 

## SEGY files

If you want to apply the seismic diffractions method on a field data, we demonstrate how to read .segy files in the notebook ```Writing_Reading_Segy_File_Example.ipynb```. You can also convert the synthetic data generated in the examples to .segy files.

## Have a question or found a problem?

Please help us improve our code!

We encourage users to open an issue in this repository when encountering any error or when in doubt about a functionality or any other subject regarding the use of the software. 

Please keep in mind that Diffrapy is mantained by few active contributors. Bug fixing might take some time to be properly adressed!
