import os
import argparse
from typing import List, Union

from nltk.tree import Tree
from nltk.grammar import CFG, Production, Nonterminal 


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--grammar_file',
    default="/home/pavlo/comp-550/comp-550-a2/input/grammar_fr.txt", 
    help='Grammar file.')
  args = parser.parse_args()

  return args


class CYK:

  def __init__(self, cfg: CFG) -> None:
    assert(isinstance(cfg, CFG)), "grammar must be of type nltk.CFG"
    self.cfg = cfg
    self.cfg_nonterminals = {p.lhs().symbol() for p in cfg.productions()}
    # self.cfg_unary_productions = {p for p in cfg.productions() 
    #                           if (p.rhs()) == 1 and 
    #                           p.rhs().sybol() in self.cfg_nonterminals}
    self.cnf = to_cnf(cfg)

  def parse(self, sentence: Union[str, List[str]], keep_cnf: bool=False) -> List[Tree]:
      words = sentence if isinstance(sentence, list) else sentence.split()

      table = [[[] for _ in range(len(words)+1)] for _ in range(len(words))]

      for j in range(1, len(words)+1):
        for production in self.cnf.productions(rhs=words[j-1]):
          table[j-1][j].append((production,))
        for i in range(j-2, -1, -1):
          for k in range(i+1, j):
            for production in self.cnf.productions():
              if production.is_nonlexical():
                B = production.rhs()[0]
                C = production.rhs()[1]
                if (B in [x[0].lhs() for x in table[i][k]] and 
                    C in [x[0].lhs() for x in table[k][j]]):
                  table[i][j].append((production, k))
      # for t in table:
      #   print(t)
      return self.build_trees(table=table, keep_cnf=keep_cnf)
  
  def build_trees(self, table: List, keep_cnf: bool) -> List:
    # check if we can retrieve at least one parse tree
    top_right_cell_lhs = [x[0].lhs() for x in table[0][-1]]
    if self.cnf.start() not in top_right_cell_lhs:
      return [] # not able to parse the sentence
    
    def move(back_ptr, i, j, pop=False):
      if len(back_ptr) == 1: # lexical production
        production = back_ptr[0]
        return Tree(
          node=production.lhs().symbol(), 
          children=[production.rhs()[0]])
      else:   # unlexical productions
        k = back_ptr[-1]

        left_back_ptrs = [x for x in table[i][k] 
                          if x[0].lhs() == back_ptr[0].rhs()[0]]
        right_back_ptrs = [x for x in table[k][j] 
                          if x[0].lhs() == back_ptr[0].rhs()[1]]
        if not pop:
          if len(right_back_ptrs) > 1 and len(right_back_ptrs[0]) > 1:
            table[k][j].remove(right_back_ptrs[0])
          elif len(left_back_ptrs) > 1 and len(left_back_ptrs[0]) > 1:
            table[i][k].remove(left_back_ptrs[0])
        subtree = Tree(
          node=back_ptr[0].lhs().symbol(),
          children=[move(left_back_ptrs[0], i, k, pop), 
                    move(right_back_ptrs[0], k, j, pop)])
                
        return subtree

    trees = []
    start = self.cnf.start()
    table[0][-1] = [x for x in table[0][-1] if x[0].lhs() == start]

    while True:
      production = table[0][-1][0]
      
      tree = Tree(
        node=start.symbol(), 
        children=move(production, 0, len(table[0])-1))
      if tree in trees:
        break
      trees.append(tree)

    # Convert from CNF to original grammar format
    if not keep_cnf:
      trees = [self.to_cfg(tree) for tree in trees]

    # keep unique trees
    unique_trees = []
    while trees:
      tree = trees.pop()
      if tree not in unique_trees:
        unique_trees.append(tree)

    return unique_trees
  
  def to_cfg(self, tree):
    # change root label to original start
    tree.set_label(self.cfg.start().symbol())

    def traverse(subtree):
      root = subtree.label()
      production = Production(
        lhs=Nonterminal(root), 
        rhs=[Nonterminal(child.label()) if isinstance(child, Tree) else child for child in subtree])
      if len(subtree) == 1: # terminal node
        if production not in self.cfg.productions():
          node = subtree.label()
          child = subtree[0]
          child_symbol = Nonterminal(child.label()) if isinstance(child, Tree) else child
          candidate = [x for x in self.cfg.productions(rhs=child_symbol) if len(x) == 1][0]
          intermediate = [x for x in self.cfg.productions(rhs=candidate.lhs()) if len(x) == 1]

          subtree = Tree(node=node, children=[Tree(node=intermediate[0].rhs()[0].symbol(), children=[child])])
          return traverse(subtree)
        return subtree
      else:     
        if production not in self.cfg.productions():
          children = []
          for child in subtree:
            label = child.label()
            if label not in self.cfg_nonterminals:
              for grandchild in child:
                children.append(grandchild)
            else:
              children.append(child)
          subtree = Tree(node=root, children=children)
        return Tree(node=root, children=[traverse(child) if isinstance(child, Tree) else child for child in subtree])

    tree = traverse(tree)
    return tree


def to_cnf(cfg: CFG) -> CFG:
  """ Takes a CFG and returns its CNF.
      Four cases are handled here:
      0. Make sure start non-terminal is never on the RHS
      1. Productions that mix terminals and non-terminals on the RHS.
      2. Productions that with a single non-terminal on the RHS.
      3. Productions with len(RHS) > 2
  """
  if cfg.is_chomsky_normal_form():
      # no modification is required
      return cfg

  # 0. Make sure start non-terminal is never on the RHS
  cfg = validate_start(cfg)
  # Case 1. Productions that mix terminals and non-terminals on the RHS.
  cfg = un_mix_rhs(cfg)
  # 2. Productions that with a single non-terminal on the RHS.
  cfg = remove_unary_productions(cfg)
  # 3. Productions with len(RHS) > 2
  cfg = make_binary_productions(cfg)

  assert cfg.is_chomsky_normal_form(), "ERROR: CFG is still not in CNF."

  return cfg


def validate_start(cfg: CFG) -> CFG:
  start = cfg.start()

  for production in cfg.productions():
    if start in production.rhs():
      new_start = Nonterminal("_ROOT_")
      new_production = Production(lhs=new_start, rhs=[start])
      cfg = CFG(
        start = new_start,
        productions = cfg.productions() + [new_production]
      )
  return cfg


def un_mix_rhs(cfg, terminal_pad:str='#') -> CFG:
  start = cfg.start()
  productions = set()

  for production in cfg.productions():
    if production.is_lexical() and len(production.rhs()) > 1: 
      # Change terminals with non-terminals and add lexical productions
      # with len(RHS) == 1
      rhs = []
      for rhs_item in production.rhs():
        if isinstance(rhs_item, Nonterminal):
          rhs.append(rhs_item)
        else:
          non_terminal = Nonterminal(symbol=terminal_pad + rhs_item.upper())
          rhs.append(non_terminal)
          productions.add(Production(lhs=non_terminal, rhs=[rhs_item])) 
      productions.add(Production(lhs=production.lhs(), rhs=rhs))
    else:
      productions.add(production)
  
  return CFG(start=start, productions=productions)


def remove_unary_productions(cfg: CFG)-> CFG:
  unary_productions = set()
  non_unary_productions = set()

  for production in cfg.productions():
    if production.is_lexical() or len(production) > 1:
      non_unary_productions.add(production)
    else:
      unary_productions.add(production)

  while unary_productions:
    unary_production = unary_productions.pop()
    unary_lhs = unary_production.lhs()
    unary_rhs = unary_production.rhs()[0]
    for production in cfg.productions(lhs=unary_rhs):
      merged_production = Production(lhs=unary_lhs, rhs=production.rhs())
  
      if merged_production.is_lexical() or len(merged_production) > 1:
        non_unary_productions.add(merged_production)
      else:
        unary_productions.add(merged_production)
  
  return CFG(cfg.start(), productions=non_unary_productions)


def make_binary_productions(cfg: CFG)-> CFG:
  valid_productions = set()
  invalid_productions = set()

  for production in cfg.productions():
    if (production.is_lexical() or (production.is_nonlexical() and len(production) == 2)):
      valid_productions.add(production)
    else:
      invalid_productions.add(production)
  
  while invalid_productions:
    invalid_production = invalid_productions.pop()
    lhs = invalid_production.lhs()
    rhs_first = invalid_production.rhs()[0]
    rhs_rest = invalid_production.rhs()[1:]
    new_non_terminal = Nonterminal(
      symbol='<' + lhs.symbol() + '-' + rhs_first.symbol() + '>')
    new_production = Production(
      lhs=invalid_production.lhs(), 
      rhs=[rhs_first, new_non_terminal])
    valid_productions.add(new_production)
    new_non_terminal_production = Production(
      lhs=new_non_terminal, rhs=rhs_rest)
    if len(new_non_terminal_production) == 2:
      valid_productions.add(new_non_terminal_production)
    else:
      invalid_productions.add(new_non_terminal_production)
  
  return CFG(start=cfg.start(), productions=valid_productions)


def main():
  args = parse_args()

  with open(args.grammar_file) as f:
    cfg = f.read()
  
  cfg = CFG.fromstring(cfg)

  cyk = CYK(cfg=cfg)

  # res = cyk.parse("I shot the elephant in my pyjamas")
  # res = cyk.parse("I shot an elephant in my pajamas")
  # res = cyk.parse("Mary saw Bob")
  # res = cyk.parse("the dog saw a man in the park")
  # res = cyk.parse("the angry bear chased the frightened little squirrel")
  # res = cyk.parse("Chatterer said Buster thought the tree was tall")
  sentences_file = "/home/pavlo/comp-550/comp-550-a2/input/sentences.txt"
  with open(sentences_file) as f:
    lines = [line.strip() for line in f.readlines() if line.strip()]

  # res = cyk.parse("le chat mange le poisson")
  for line in lines:
    res = cyk.parse(line)
    
    print(f"{line} : {len(res)}")

  # for r in res:
  #   r.pretty_print()

if __name__ == "__main__":
  main()