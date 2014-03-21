This directory tests various ways of copying memory between the CPU
and GPU in an OpenCL program. The goal is to determine how much time
is spend on bandwidth and how much is spent on latency. My experience
is that bandwidth time dominates very quickly, but the goal is to find
some data to support that.
