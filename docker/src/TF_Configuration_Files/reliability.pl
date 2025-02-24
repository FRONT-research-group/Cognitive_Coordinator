flavor(low_reliability, 1.0, 40.0, 0.5, 512).
flavor(medium_reliability, 41.0, 80.0, 1.0, 1024).
flavor(high_reliability, 81.0, 100.0, 2.0, 2048).

assign_flavor(Score, CPU, Memory, Flavor) :-
    flavor(Flavor, Min, Max, ReqCPU, ReqMem),
    Score >= Min, Score =< Max,
    CPU >= ReqCPU, Memory >= ReqMem.

fallback_flavor(CPU, Memory, Flavor) :-
    flavor(Flavor, _, _, ReqCPU, ReqMem),
    CPU >= ReqCPU, Memory >= ReqMem.

flavor_upper_bound(Flavor, MaxScore) :-
    flavor(Flavor, _, MaxScore, _, _).