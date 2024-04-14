
        set terminal png
        set output 'speedup.png'
        set title 'Speed Up vs Number of Processing Elements'
        set xlabel 'Number of Processing Elements (p)'
        set ylabel 'Speed Up'
        set grid
        plot 'speedup.dat' using 1:2:3 with lines
    