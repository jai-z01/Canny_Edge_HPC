
        set terminal png
        set output 'parallelefficiency.png'
        set title 'Parallel Efficiency vs Number of Processing Elements'
        set xlabel 'Number of Processing Elements (p)'
        set ylabel 'Parallel Efficiency (%)'
        set zlabel 'Problem Size (N)'
        set grid
        splot 'parallelefficiency.dat' using 1:2:3 with lines
    