STARTING!
Duplicate flags: num_threads
Min num runs: [50]
Min runs duration (seconds): [1]
Max runs duration (seconds): [150]
Inter-run delay (seconds): [-1]
Num threads: [1]
Benchmark name: []
Output prefix: []
Min warmup runs: [1]
Min warmup runs duration (seconds): [0.5]
Graph: [/data/local/tmp/mnv3_seg_float.tflite]
Input layers: []
Input shapes: []
Input value ranges: []
Input layer values files: []
Use legacy nnapi : [0]
Allow fp16 : [0]
Require full delegation : [0]
Enable op profiling: [1]
Max profiling buffer entries: [1024]
CSV File to export profiling data to: []
Enable platform-wide tracing: [0]
#threads used for CPU inference: [1]
Max number of delegated partitions : [0]
Min nodes per partition : [0]
External delegate path : []
External delegate options : []
Use gpu : [1]
Allow lower precision in gpu : [1]
Enable running quant models in gpu : [1]
Use Hexagon : [0]
Hexagon lib path : [/data/local/tmp]
Hexagon Profiling : [0]
Use nnapi : [0]
Use xnnpack : [0]
Loaded model /data/local/tmp/mnv3_seg_float.tflite
Applied GPU delegate, and the model graph will be completely executed w/ the delegate.
The input model file size (MB): 3.79893
Initialized session in 1451.51ms.
Running benchmark for at least 1 iterations and at least 0.5 seconds but terminate if exceeding 150 seconds.
count=56 first=11147 curr=8146 min=7055 max=12036 avg=8779.68 std=1115

Running benchmark for at least 50 iterations and at least 1 seconds but terminate if exceeding 150 seconds.
count=110 first=8325 curr=10016 min=7863 max=10688 avg=8680.33 std=591

Inference timings in us: Init: 1451507, First inference: 11147, Warmup (avg): 8779.68, Inference (avg): 8680.33
Note: as the benchmark tool itself affects memory footprint, the following is only APPROXIMATE to the actual memory footprint of the model at runtime. Take the information at your discretion.
Peak memory footprint (MB): init=63.4883 overall=63.4883
Profiling Info for Benchmark Initialization:
============================== Run Order ==============================
	             [node type]	          [start]	  [first]	 [avg ms]	     [%]	  [cdf%]	  [mem KB]	[times called]	[Name]
	 ModifyGraphWithDelegate	            0.000	 1450.744	 1450.744	 99.998%	 99.998%	 62364.000	        1	ModifyGraphWithDelegate/0
	         AllocateTensors	         1450.830	    0.028	    0.016	  0.002%	100.000%	     0.000	        2	AllocateTensors/0

============================== Top by Computation Time ==============================
	             [node type]	          [start]	  [first]	 [avg ms]	     [%]	  [cdf%]	  [mem KB]	[times called]	[Name]
	 ModifyGraphWithDelegate	            0.000	 1450.744	 1450.744	 99.998%	 99.998%	 62364.000	        1	ModifyGraphWithDelegate/0
	         AllocateTensors	         1450.830	    0.028	    0.016	  0.002%	100.000%	     0.000	        2	AllocateTensors/0

Number of nodes executed: 2
============================== Summary by node type ==============================
	             [Node type]	  [count]	  [avg ms]	    [avg %]	    [cdf %]	  [mem KB]	[times called]
	 ModifyGraphWithDelegate	        1	  1450.744	    99.998%	    99.998%	 62364.000	        1
	         AllocateTensors	        1	     0.032	     0.002%	   100.000%	     0.000	        2

Timings (microseconds): count=1 curr=1450776
Memory (bytes): count=0
2 nodes observed



Operator-wise Profiling Info for Regular Benchmark Runs:
============================== Run Order ==============================
	             [node type]	          [start]	  [first]	 [avg ms]	     [%]	  [cdf%]	  [mem KB]	[times called]	[Name]
	     TfLiteGpuDelegateV2	            0.000	    8.317	    8.671	100.000%	100.000%	     0.000	        1	[Identity]:77

============================== Top by Computation Time ==============================
	             [node type]	          [start]	  [first]	 [avg ms]	     [%]	  [cdf%]	  [mem KB]	[times called]	[Name]
	     TfLiteGpuDelegateV2	            0.000	    8.317	    8.671	100.000%	100.000%	     0.000	        1	[Identity]:77

Number of nodes executed: 1
============================== Summary by node type ==============================
	             [Node type]	  [count]	  [avg ms]	    [avg %]	    [cdf %]	  [mem KB]	[times called]
	     TfLiteGpuDelegateV2	        1	     8.671	   100.000%	   100.000%	     0.000	        1

Timings (microseconds): count=110 first=8317 curr=10006 min=7854 max=10679 avg=8671.49 std=590
Memory (bytes): count=0
1 nodes observed



