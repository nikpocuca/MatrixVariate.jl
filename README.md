# MatrixVariate.jl

A complete statistical framework for analyzing matrix variate data.

## Installation
To download the package simply run the following in the pkg mode within the REPL.
```julia
add https://github.com/nikpocuca/MatrixVariate.jl
```

## Includes the following:

### MatrixNormTest
A framework for assessing matrix variate normality.

![](/docs/src/src/norm.png)
![](/docs/src/src/nnorm.png)

### Bilinear Factor Analyzers
A mixture of Factor Analyzers for matrices. [arXiv paper](https://arxiv.org/abs/1712.08664).

Mean of each group (5 and 7) for MNIST results. 

![](5s.png)
![](7s.png)

### Skewed Family of Bilinear Factor Analyzers

A mixture of Factor Analyzers for matrices with a family of four skewed matrix variate distributions.  [arXiv paper](https://arxiv.org/abs/1809.02385)

- Matrix Variate Skewed-t Distribution 
- Matrix Variate Variance Gamma
- Matrix Variate Normal Inverse Gaussian
- Matrix Variate Generalized Hyperbolic 

## Citing  MatrixVariate.jl

To cite MatrixVariate.jl, please reference the bibtex below:

```

@Manual{pocuca19,
 	title = {MatrixVariate.jl: A complete statistical framework for analyzing matrix variate data},
  	author = {Nikola Po\v{c}u\v{c}a and Michael P. B. Gallaugher and Paul D. McNicholas},
  	year = {2019},
  	note = {julia package version 0.2.0},
	URL={http://github.com/nikpocuca/MatrixVariate.jl}
}

```
