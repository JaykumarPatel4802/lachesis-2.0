# Intel CPU Energy Measurements

### Intel RAPL (Running Average Power Limit)
RAPL (Running Average Power Limit) is a feature found in modern processors that enable power monitoring and management. There are several domains in RAPL, each one representing a different part of the processor where power consumption can be measured and controlled. The main domains are:

1. PKG (Package): The entire processor package, which includes all cores, cache, integrated GPU (if present), and other components on the same physical chip.
2. PP0 (Platform Power Plane 0): This domain represents the power consumption of the cores (CPU cores) in the processor.
3. PP1 (Platform Power Plane 1): Represents the power consumption of the integrated GPU, if the processor has one.
4. DRAM (Dynamic Random Access Memory): Represents the power consumption of the memory modules used by the processor.
5. PSYS (Platform System): Represents the overall system power consumption, including non-processor components such as motherboard and peripherals.

For our work, we use the Intel(R) Xeon(R) Gold 6240R CPU @ 2.40GHz, a server-grade processor based on the CascadeLake Architecture, which doesn't provide PP0 and PP1 domains, only PKG and DRAM. 

### Likwid-Powermeter

There are several tools that we can use to get power measurements (i.e., Powerstat, PowerTop, Perf, Likwid). We found that likwid-powermeter is very straightfoward to install and use and provides a rich set of energy AND power information. Likwid-powermeter uses the RAPL interface to fetch energy and power measurements from different domains of CPU. One important feature is we can get the power and energy consumption for particular processes. 

Installing likwid-powermeter
```
git clone https://github.com/RRZE-HPC/likwid.git
cd likwid
make
sudo make install
which likwid-powermeter # should see ...
```

Running likwid-powermeter
```
likwid-powermeter python3 matmult.py 10000
```

Output will be:
```
-----------------------------------------------------
CPU name:	Intel(R) Xeon(R) Gold 6240R CPU @ 2.40GHz
CPU type:	Intel Cascadelake SP processor
CPU clock:	2.39 GHz
-----------------------------------------------------
Runtime: 7.9624 s
Measure for socket 0 on CPU 0
Domain PKG:
Energy consumed: 521.537 Joules
Power consumed: 65.5001 Watt
Domain PP0:
Energy consumed: 0 Joules
Power consumed: 0 Watt
Domain DRAM:
Energy consumed: 60.3582 Joules
Power consumed: 7.58041 Watt
Domain PLATFORM:
Energy consumed: 0 Joules
Power consumed: 0 Watt

Measure for socket 1 on CPU 1
Domain PKG:
Energy consumed: 499.67 Joules
Power consumed: 62.7538 Watt
Domain PP0:
Energy consumed: 0 Joules
Power consumed: 0 Watt
Domain DRAM:
Energy consumed: 71.0124 Joules
Power consumed: 8.91847 Watt
Domain PLATFORM:
Energy consumed: 0 Joules
Power consumed: 0 Watt
-----------------------------------------------------
```
Note that power and energy consumed for domains PP0 are both 0 because our processor does not provide this domain for energy collection.

### Using Our Scripts
Super easy:
1. Run the `./[app]/energy-[app].sh`
2. Data is output in two csv files: `./[app]/data-energy-[app].csv` and `./[app]/data-util-[app].csv`

Then, you can easily join the data on the `datatime`, `input_size`, `cpu_level`.


