
def render_graph_svg(component_id, graph, posns, labels=None, hide_virtual=True):
    if component_id is None:
        component_id = graph.name
    if labels is None:
        labels = { g:g for g in graph.nodes }
    node_circle = """<circle class="stroke_topaz fill_lightblue stroke_width_thin fill_alpha_25pc" cx=\"{cx}\" cy=\"{cy}\" r=\"{r}\" /> """
    open_circle = """<circle class="stroke_black fill_white stroke_width_thin" cx=\"{cx}\" cy=\"{cy}\" r=\"{r}\" /> """
    virtual_node_circle = """<circle class="stroke_green fill_avocado stroke_width_thin" cx=\"{cx}\" cy=\"{cy}\" r=\"{r}\" /> """
    cext_t = """<text x="{cx}" y="{cy}" text-anchor="middle" alignment-baseline="middle">{text}</text>"""
    nodes_svg = ""
#    edge_line = """<line class="stroke_black stroke_width_thin" x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" marker-end="url(#arrow)"/>"""
    edge_line = """<path class="stroke_black stroke_width_thin" d="M{x1} {y1} L{x2} {y2}" marker-end="url(#arrow)"/>"""
    for edge in graph.edges:
        if ("__virtual" not in str(edge[1]) or not hide_virtual):
            edge_start_x, edge_start_y, edge_end_x, edge_end_y = posns[edge[0]][0], posns[edge[0]][1], posns[edge[1]][0], posns[edge[1]][1]
            #nodes_svg = nodes_svg + edge_line.format(x1=edge_start_x, y1=edge_start_y, x2=edge_end_x, y2=edge_end_y )
            x_diff, y_diff = (edge_end_x-edge_start_x), (edge_end_y-edge_start_y)
            line_dist = (x_diff**2 + y_diff**2)**(1/2)
            radius_offset = 1-(0.4/line_dist)
            nodes_svg = nodes_svg + edge_line.format(x1=edge_start_x, y1=edge_start_y,
                                                     x2=edge_start_x+(x_diff*radius_offset), y2=edge_start_y+(y_diff*radius_offset) )

        else:
            pass


    for node, pos in posns.items():
        if "__virtual" not in str(node) :
            nodes_svg = nodes_svg + node_circle.format(r=0.3, cx=pos[0], cy=pos[1])
            nodes_svg = nodes_svg + cext_t.format(cx=pos[0], cy=pos[1], text=str(labels.get(node, "_")))
        elif not hide_virtual:
            nodes_svg = nodes_svg + virtual_node_circle.format(r=0.3, cx=pos[0], cy=pos[1])
            nodes_svg = nodes_svg + cext_t.format(cx=pos[0], cy=pos[1], text=str(node[-3:]))

    return  """<g id="{component_id}"><g>{contents}</g></g>""".format(component_id=component_id, contents=nodes_svg)

def render_nodes_svg(graph, posns, labels=None, nodes_l=None, styles=None, hide_virtual=True):

    if nodes_l is None:
        nodes_l = [k for k in graph.nodes.keys()]

    if hide_virtual:
        nodes_l = [l for l in nodes_l if "__virtual" not in l]

    if labels is None:
        labels = { g:g for g in nodes_l }

    for l in nodes_l:
        if l not in labels:
            labels[l] = l

    if styles is None:
        if hide_virtual:
            styles = { k : SVGTemplate("CIRCLE") for k in nodes_l}
        else:
            styles = { k : SVGTemplate("RECT") if "__virtual" in k else SVGTemplate("CIRCLE") for k in nodes_l}
    else:
        for k in nodes_l:
            if k not in styles:
                styles[k]=SVGTemplate("CIRCLE")

    nodes_svg=""

    for node in nodes_l:
        if all([p in styles[node].parameters.keys() for p in ("cx", "cy") ]) :
            nodes_svg = nodes_svg + styles[node].svg(cx=posns[node][0], cy=posns[node][1])
        elif all([p in styles[node].parameters.keys() for p in ("x", "y", "width", "height") ]) :
            # Convert central x/y to bottom-left x/y
            x = posns[node][0] - (styles[node].parameters["width"]/2)
            y = posns[node][1] - (styles[node].parameters["height"]/2)
            nodes_svg = nodes_svg + styles[node].svg(x=x, y=y)
        else:
            nodes_svg = nodes_svg + styles[node].svg(x=posns[node][0], y=posns[node][1])
    return nodes_svg


def render_labels_svg(graph, posns, labels=None, nodes_l=None, styles=None, hide_virtual=True):

    if nodes_l is None:
        nodes_l = [k for k in graph.nodes.keys()]

    if hide_virtual:
        nodes_l = [l for l in nodes_l if "__virtual" not in l]

    if labels is None:
        labels = { g:g for g in nodes_l }

    for l in nodes_l:
        if l not in labels:
            labels[l] = l

    if styles is None:
        styles = { k : SVGTemplate("TEXT") for k in nodes_l}
    else:
        for k in nodes_l:
            if k not in styles:
                styles[k]=SVGTemplate("TEXT")

    nodes_svg=""

    for node in nodes_l:
        if all([p in styles[node].parameters.keys() for p in ("cx", "cy") ]) :
            nodes_svg = nodes_svg + styles[node].svg(cx=posns[node][0], cy=posns[node][1], text=labels[node])
    return nodes_svg



def render_edges_svg(graph, posns, edges_l=None, styles=None, hide_virtual=True):

    if edges_l is None:
        edges_l = [k for k in graph.edges.keys()]

    if hide_virtual:
        edges_l = [l for l in edges_l if ("__virtual" not in l[0] and "__virtual" not in l[1])]

    if labels is None:
        labels = { g:str(g[0]) + "-> " + str(g[1]) for g in edges_l }

    for l in edges_l:
        if l not in labels:
            edges_l[l] = str(l[0]) + "-> " + str(l[1])

    if styles is None:
        styles = { k : SVGTemplate("ARROW") for k in edges_l}
    else:
        for k in nodes_l:
            if k not in styles:
                styles[k]=SVGTemplate("LINE")

    nodes_svg=""

    for edge in edges_l:
        if all([p in styles[edge].parameters.keys() for p in ("x1", "x2", "y1", "y2") ]) :
            nodes_svg = nodes_svg + styles[edge].svg(x1=posns[edge[0]][0], y1=posns[edge[0]][1],x2=posns[edge[1]][0], y2=posns[edge[1]][1])
    return nodes_svg

class SVGTemplate(object):

    def __init__(self, style, **parameters):

        self._style_mapping={ "CIRCLE" : self._circle_style,
                              "RECT"   : self._rect_style,
                              "LINE"   : self._line_style,
                              "ARROW"  : self._arrow_style,
                              "TEXT"  : self._text_style
         }

        if str(style).upper() in self._style_mapping:
            self.style=style
            self.style_func=self._style_mapping[str(style).upper()]
            self.style_func()
        else:
            raise ValueError("{s} is not a recognised style.")
        # Update provided parameters, overriding any defaults
        for k,v in self.parameters.items():
            if k in parameters:
                self.parameters[k]=parameters[k]

    def svg(self, **parameters):
        run_parms={}
        # Override any provided parameters
        for k,v in self.parameters.items():
            if k in parameters:
                run_parms[k]=parameters[k]
            else:
                run_parms[k]=self.parameters[k]

        # Perform list to string conversion
        run_parms["classes"]=" ".join([c for c in run_parms["classes"]])

        return self.template.format(**run_parms)

    def __repr__(self):
        if len(self.parameters)>0:
            kwas= "," + ",".join([k + "=" + str(v) for k,v in self.parameters.items()])
        else:
            kwas = ""
        return str(self.__class__.__name__) + "('" + self.style + "'" + kwas + ")"

    def _circle_style(self):
        self.template="""<circle class="{classes}" cx="{cx}" cy="{cy}" r="{r}" />"""
        self.parameters={"classes": ["stroke_black", "fill_white", "stroke_width_thin"], "cx":0, "cy":0, "r":1}

    def _rect_style(self):
        self.template="""<rect class="{classes}" x="{x}" y="{y}" width="{width}" height="{height}" />"""
        self.parameters={"classes": ["stroke_black", "fill_white", "stroke_width_thin"], "x":0, "y":0, "width":1, "height":1}

    def _line_style(self):
        self.template="""<path class="{classes}" d="M{x1} {y1} L{x2} {y2}" />"""
        self.parameters={"classes": ["stroke_black", "stroke_width_thin"], "x1":0, "y1":0, "x2":1, "y2":1 }

    def _arrow_style(self):
        self.template="""<path class="{classes}" d="M{x1} {y1} L{x2} {y2}" marker-end="url(#arrow)"/>"""
        self.parameters={"classes": ["stroke_black", "stroke_width_thin"], "x1":0, "y1":0, "x2":1, "y2":1 }

    def _text_style(self):
        self.template="""<text x="{cx}" y="{cy}" text-anchor="middle" alignment-baseline="middle">{text}</text>"""
        self.parameters={"classes": [], "cx":0, "cy":0, "text":"None" }

def svg_base(width_height=(500,500), viewbox=(0,0,10,10), contents=None):
    viewbox = ", ".join([str(v) for v in viewbox])
    width, height = width_height
    defs = """<defs>
        <marker id="arrow" markerWidth="1" markerHeight="1" refX="0", refY="3" orient="auto" overflow="visible" markerUnits="strokeWidth">
        <path d="M-7,0 L-7,6 L0,3 z" fill="#fff" stroke="black" fill-opacity="1.0"/>
        </marker>
        </defs>"""
    svg_chunk = """<svg version="1.1" xmlns="http://www.w3.org/2000/svg"

    xmlns:xlink="http://www.w3.org/1999/xlink" x="0" y="0"
    width="{width}" height="{height}" viewBox="{viewbox}">
    {defs}
      {contents}
      </svg >
        """.format(width=width, height=height, viewbox=viewbox, defs=defs, contents=contents)
    return svg_chunk



def css_style():
    css_style = """
        <style>
            text {font-size:0.03em;}
            .small_text {font-size:0.01em;}
            .fill_white { fill: white; }
            .fill_black { fill: black; }
            .fill_red { fill: #FF3333; }
            .fill_orange { fill: #FFAA00; }
            .fill_yellow { fill: #FFFF00; }
            .fill_green { fill: #00FF00; }
            .fill_avocado { fill: #9bd975; }
            .fill_cyan { fill: #00FFFF; }
            .fill_lightblue { fill: #00AAFF; }
            .fill_blue { fill: #0000FF; }
            .fill_indigo { fill: #4400FF; }
            .fill_violet { fill: #AA00FF; }
            .fill_none { fill: none;}

            .fill_garnet { fill: #ca2020;}
            .fill_buttercup { fill: #dbac01;}
            .fill_jade { fill: #05776a;}
            .fill_topaz { fill: #246bcd;}

            .fill_alpha_none { fill-opacity: 0.0;}
            .fill_alpha_25pc { fill-opacity: 0.25;}
            .fill_alpha_50pc { fill-opacity: 0.5;}
            .fill_alpha_100pc { fill-opacity: 1.0;}

            .stroke_white { stroke: white; }
            .stroke_black { stroke: black; }
            .stroke_red { stroke: #FF3333; }
            .stroke_orange { stroke: #FFAA00; }
            .stroke_yellow { stroke: #FFFF00; }
            .stroke_green { stroke: #00FF00; }
            .stroke_avocado { stroke: #9bd975; }
            .stroke_cyan { stroke: #00FFFF; }
            .stroke_lightblue { stroke: #00AAFF; }
            .stroke_blue { stroke: #0000FF; }
            .stroke_indigo { stroke: #4400FF;}
            .stroke_violet { stroke: #AA00FF; }
            .stroke_garnet { stroke: #ca2020;}
            .stroke_buttercup { stroke: #dbac01; }
            .stroke_jade { stroke: #05776a; }
            .stroke_topaz { stroke: #246bcd; }
            .stroke_none { stroke: none;}

            .stroke_width_fine { stroke-width:0.03; }
            .stroke_width_thin { stroke-width:0.05; }
            .stroke_width_mid { stroke-width:0.1; }
            .stroke_width_wide { stroke-width:0.5; }

        </style>"""
    return css_style
