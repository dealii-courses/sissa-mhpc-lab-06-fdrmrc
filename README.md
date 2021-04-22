#  Lab 06 - A posteriori error estimation and adaptive FEM
## Theory and Practice of Finite Elements

**Luca Heltai** <luca.heltai@sissa.it>

* * * * *

## General Instructions

For each of the point below, extend the `Poisson` class with functions that
perform the indicated tasks, trying to minimize the amount of code you copy and
paste, possibly restructuring existing code by adding arguments to existing
functions, and generating wrappers similar to the `run` method (e.g.,
`run_exercise_3`).

Once you created a function that performs the given task, add it to the
`poisson-tester.cc` file, and make sure all the exercises are run through the
`gtest` executable, e.g., adding a test for each exercise, as in the following
snippet:

```C++
TEST_F(PoissonTester, Exercise3) {
   run_exercise_3();
}
```

By the end of this laboratory, you will have modified your Poisson code to
allow also non-homogeneous Neumann boundary conditions on different parts of
the domain, and you will have added some more options to the solver, enabling
usage of a direct solver, or of some more sofisticated preconditioners.

## Lab-06
### step-6

1.  See documentation of step-6 at
    <https://www.dealii.org/current/doxygen/deal.II/step_6.html>

2. Add the parameters
   
    - `Mapping degree`
    - `Marking strategy`
    - `Estimator type`
    - `Coarsening and refinement factors`
   
where `Mapping degree` controls the degree of the mapping used in the code,
`Marking strategy` is a choice between `global|fixed_fraction|fixed_number`,
`Estimator type` is a choice between `exact|kelly|residual`, and
`Coarsening and refinement factors` is a `std::pair<double, double>` containing
the arguments to pass to the `GridRefinement::refine_and_coarsen_fixed_*`
functions

1. Make sure all `FEValues` classes use a mapping with the correct order, and
make sure you use the correct mapping in the output as well (if you have a
recent Paraview)

4. Add a `Vector<float` field `error_per_cell` to `Poisson`, to be filled by
the method `estimate`

5. Add a method `residual_error_estimator` that computes the residual error estimator, using `FEInterfaceValues` and `FEValues`

6. Add a method `estimate` to the `Poisson` class, to compute the H1 seminorm
of the difference between the exact and computed solution if `Estimator type`
is `exact`, calls `KellyErrorEstimator<dim>::estimate` if `Estimator type` is
`kelly`, and calls `residual_error_estimator` if `Estimator type` is `residual`

7. Add to the convergence tables also the estimator you computed. This should be
identical to the `H1` semi-norm in the case where `Estimator type` is `exact`

8. Taking the exact solution computed on the L-shaped domain, compute the rate
at which the adaptive finite element method converges in terms of the number of
degrees of freedom using the three estimators above

9. Set zero boundary conditions, forcing term equal to four, exact solution
equal to `-x^2-y^2+1` and solve the problem on a circle with center in the
origin and radius one, for various finite element and mapping degrees. What do
you observe when the mapping degree does not match the finite element degree?
How do you explain this?

10. Create a test that reproduces exactly the behaviour of `step-6`, using only
your parameter file