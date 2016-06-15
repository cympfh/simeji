set terminal png
set output 'matrix.png'

set palette gray negative

set size ratio 1

unset xtics
set yrange [] reverse

set tics scale 0,0.001
set x2tics 10
set ytics 10
set autoscale x2fix
set autoscale xfix
set autoscale y2fix
set autoscale yfix

# separating lines
set grid front mx2tics mytics lw 2.0 lt -1 lc rgb 'white'

set x2label ''
set ylabel ''

plot 'mnist.out.dat' matrix w image notitle
