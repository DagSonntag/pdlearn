# pdlearn
## Motivation
pdlearn is a python package that facilitates easier handling of sklearn transformation methods when using pandas 
DataFrames. This is done by wrapping the existing sklearn transformers with pdlearn code such that they take both 
pandas DataFrames in and also return the data in such format. Normal sklearn methods can usually take 
dataframes in, but then mostly converts them to numpy-arrays to improve performance. This makes sense in many use-cases,
but for exploration work it is often useful to keep the dataframe structure throughout the Pipelines.

## Scope and goal
This package is only a development package and meant to be a storage for these kind of methods to avoid reimplementing 
them every time. It is by no means complete, and transformer classes are only implemented on demand. Once implemented
they are tested on the data available, but no regression or more advanced testing is used. Moreover, the functionality 
of the transformer classes may also not be completely implemented (like for example inverse transforms), if it is not 
required for the development use-case.

Hence, use only as a goto example for code and always check the code of the transformation you intend to use before 
doing so to make sure it handles your use-case.

## Structure 
The structure is created to follow the sklearn package as close as possible to help users and developers manoeuvre the
different subpackages




