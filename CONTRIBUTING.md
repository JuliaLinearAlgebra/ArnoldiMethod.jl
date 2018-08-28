# How to contribute to ArnoldiMethod.jl

## Did you find a bug?
[Open an issue](https://github.com/haampie/ArnoldiMethod.jl/issues) on GitHub 
and share a minimal, reproducible example.

## How can I contribute?
If you're interested in numerical linear algebra and want to improve this 
package, make sure you skim through some literature:

- Templates for the Solution of Algebraic Eigenvalue Problems available on 
  Netlib: in particular the [Implicitly Restarted Arnoldi Method](http://www.netlib.org/utk/people/JackDongarra/etemplates/node220.html)
  chapter is interesting.

If you're unfamiliar with Julia, the documentation is an excellent starting
point:

- https://docs.julialang.org/en/v1/

A recommended workflow to contribute is as follows:

1. Fork the package on GitHub;
2. Open Julia's REPL and hit `]` to enter the package manager mode;
3. Run
   ```julia
   (v1.0) pkg> dev git git@github.com:YOURUSERNAME/ArnoldiMethod.jl.git
   ```
4. Run the tests locally to make sure things work:
   ```julia
   (v1.0) pkg> test ArnoldiMethod
   ```
5. Make changes to the code;
6. Push the changes to your forked repository;
7. Open a [pull request](https://github.com/haampie/ArnoldiMethod.jl/pulls).