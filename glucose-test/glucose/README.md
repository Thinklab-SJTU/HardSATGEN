# The Glucose SAT Solver

This a CMake Version of the [glucose sat solver 4.0](http://www.labri.fr/perso/lsimon/glucose/).

> Glucose is based on a new scoring scheme (well, not so new now, it was introduced in 2009) for the clause learning mechanism of so called "Modern" SAT sovlers (it is based our IJCAI'09 paper). It is designed to be parallel, since 2014. This page summarizes the techniques embedded in all the versions of glucose. The name of the Solver name is a contraction of the concept of "glue clauses", a particular kind of clauses that glucose detects and preserves during search.
> Glucose is heavily based on Minisat, so please do cite Minisat also if you want to cite Glucose.
> 
> -- Gilles Audemard and Laurent Simon 

```
Directory overview:
==================

mtl/            Minisat Template Library
core/           A core version of the solver glucose (no main here)
simp/           An extended solver with simplification capabilities
parallel/       A multicore version of glucose
README
LICENSE
Changelog
```

## To build 

```
mkdir build
cd build
cmake ..
make 
``` 
