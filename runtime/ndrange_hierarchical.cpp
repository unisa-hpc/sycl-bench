TODO 2 SYCL codes implementing the same things with either ndrange or hierarchical parallel for, to show the difference the potential runtime optimization
Note: Segmented reduction has already hierarchical and ndrange implementations - if possible I would suggest having those two variants wherever possible, since ndrange on triSYCL and hipSYCL CPU backends will be prohibitively slow.
