# Utility functions for working with 2d xy dictionaries where positions are
# keyed by some identifier, and recorded as xy tuples.

def rescale_positions(positions_d, xy_bounds):
    minx, maxx, miny, maxy = xy_bounds
    xvals, yvals = zip(*[v for v in positions_d.values()])
    minxvals, maxxvals, minyvals, maxyvals = min(xvals), max(xvals), min(yvals), max(yvals)
    xrangvals, yrangvals = maxxvals-minxvals, maxyvals-minyvals
    xrang, yrang = maxx-minx, maxy-miny
    if xrangvals == 0:
        xrangvals = 1
    if yrangvals == 0:
        yrangvals = 1

    return { k : ((((v[0]-minxvals)/xrangvals)*xrang)+minx, (((v[1]-minyvals)/yrangvals)*yrang)+miny) for k,v in positions_d.items() }


def translate_positions(positions_d, xy_trans):
    return { k : (v[0]+xy_trans[0], v[1]+xy_trans[1]) for k,v in positions_d.items()}


def jiggle_positions(positions_d, jiggle_amount):
    ja = jiggle_amount
    return { k: (v[0]+((random()*ja)-(ja/2)), v[1]+((random()*ja)-(ja/2)) ) for k,v in positions_d.items() }


def snap_to_lattice(positions_d, width=None, height=None):

    if width is None:
        width=len(positions_d)

    if height is None:
        height=len(positions_d)

    x_vals, y_vals = [xy for xy in zip(*[xy for xy in positions_d.values()])]
    minx,miny=min(x_vals), min(y_vals)
    x_vals = [x-minx for x in x_vals]
    y_vals = [y-miny for y in y_vals]

    maxx,maxy= max(x_vals), max(y_vals)
    xstep,ystep = maxx/(width-1), maxy/(height-1)
    lattice_x = [(xstep * e) - (xstep/2) for e in range(0,width+1)]
    lattice_y = [(ystep * e) - (ystep/2) for e in range(0,height+1)]
    snap_xy_d = {}
    for e,(k,p) in enumerate(locs_xy_d.items()):

        snap_xy_d[k]=([(x_vals[e])>l for l in lattice_x].index(False)-1,
                      [(y_vals[e])>l for l in lattice_y].index(False)-1 )
    return snap_xy_d
