import digraph
import random
import uuid

ALPHABET_UC = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
GREEK_UC = list("ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ")
GREEK_LC = list("αβγδεζηθικλμνξοπρστυφχψω")

def gen_name_sequence(alphabet=ALPHABET_UC):
    e=-1
    finished=False
    while not finished:
        e=e+1
        l,r = e % len(alphabet), e // len(alphabet)
        #print(l,r)
        if e<len(alphabet):
            yield ( alphabet[l])
        elif e<(len(alphabet)+1)*len(alphabet):
            yield alphabet[r-1]+alphabet[l]
        elif e<(len(alphabet)**3)+(len(alphabet)**2)+(len(alphabet)):
            p,q = (r-1) % len(alphabet), (r-1) // len(alphabet)
            yield ( alphabet[q-1]+alphabet[p]+alphabet[l])
        else:
            finished = True

def gen_uuid():
    finished=False
    while not finished:
        yield str(uuid.uuid4())

def gen_random_name(length=8, alphabet=ALPHABET_UC):
    finished=False
    while not finished:
        yield "".join([random.choice(alphabet) for r in range(0,length)])


def random_dag(nodesize=15, tree_factor=1, connected_factor=1, alphabet=ALPHABET_UC):
    rdag = digraph.DiGraph()
    node_collection = set()
    name_generator=gen_name_sequence(alphabet=alphabet)
    for n in range(0,nodesize):
        node_name=next(name_generator)
        rdag.add_node(node_name)

        tree_roll = random.random()
        connected_roll = random.random()
        if connected_roll < connected_factor:
            if len(node_collection)>0:
                extant_node = random.choice([n for n in node_collection])
                rdag.add_edge((extant_node, node_name))
            else:
                rdag.add_node(node_name)

        else:
            # This node is isolated with no linking nodes
            pass

        if tree_roll > tree_factor and len(node_collection)>0:
            try:
                extant_node = random.choice([n for n in node_collection if n not in rdag.ancestors(node_name)])
                rdag.add_edge((extant_node, node_name))
            except Exception as e:
                print (e)

        node_collection.add(node_name)

    return rdag
