# Wumpus World - knowledge-based agent (logical inference over percepts)
from collections import deque

SIZE = 4
DIRS = {'N': (0, 1), 'E': (1, 0), 'S': (0, -1), 'W': (-1, 0)}
LEFT = {'N': 'W', 'W': 'S', 'S': 'E', 'E': 'N'}
RIGHT = {v: k for k, v in LEFT.items()}


def neighbours(c):
    x, y = c
    return [(x + dx, y + dy) for dx, dy in DIRS.values()
            if 1 <= x + dx <= SIZE and 1 <= y + dy <= SIZE]


class World:
    def __init__(self, wumpus, gold, pits):
        self.wumpus, self.gold, self.pits = wumpus, gold, set(pits)

    def percept(self, c):
        nb = neighbours(c)
        stench = 'Stench' if self.wumpus in nb else None
        breeze = 'Breeze' if any(p in nb for p in self.pits) else None
        glitter = 'Glitter' if c == self.gold else None
        return [stench, breeze, glitter, None, None]   # [Stench, Breeze, Glitter, Bump, Scream]


class Agent:
    def __init__(self):
        self.pos, self.face = (1, 1), 'E'
        self.visited, self.no_pit, self.no_wumpus = set(), set(), set()
        self.stench_at, self.breeze_at = set(), set()
        self.actions, self.has_gold = 0, False

    # ---- knowledge base update (TELL) ----
    def tell(self, c, p):
        self.visited.add(c)
        self.no_pit.add(c); self.no_wumpus.add(c)
        if p[0]: self.stench_at.add(c)
        else: self.no_wumpus.update(neighbours(c))
        if p[1]: self.breeze_at.add(c)
        else: self.no_pit.update(neighbours(c))

    # ---- inference (ASK) ----
    def safe(self, c):
        return c in self.no_pit and c in self.no_wumpus

    def wumpus_cell(self):
        cands = None
        for s in self.stench_at:
            n = {x for x in neighbours(s) if x not in self.no_wumpus}
            cands = n if cands is None else cands & n
        return next(iter(cands)) if cands and len(cands) == 1 else None

    def possible_pits(self):
        return {n for b in self.breeze_at for n in neighbours(b) if n not in self.no_pit}

    # ---- planning ----
    def path(self, goal):
        q, seen = deque([(self.pos, [])]), {self.pos}
        while q:
            c, p = q.popleft()
            if c == goal: return p
            for n in neighbours(c):
                if n not in seen and (n in self.visited or self.safe(n)):
                    seen.add(n); q.append((n, p + [n]))
        return None

    def choose_target(self):
        hz = self.possible_pits()
        cands = [c for c in self.no_pit & self.no_wumpus if c not in self.visited]
        cands = [c for c in cands if self.path(c)]
        if not cands: return None
        return min(cands, key=lambda c: (len(self.path(c)),
                                         sum(n in hz for n in neighbours(c)), -c[0]))

    def step_to(self, c, log):
        dx, dy = c[0] - self.pos[0], c[1] - self.pos[1]
        want = [d for d, v in DIRS.items() if v == (dx, dy)][0]
        while self.face != want:
            if LEFT[self.face] == want or LEFT[LEFT[self.face]] == want and self.face in 'NS':
                self.face = LEFT[self.face]; log.append('Turn Left')
            else:
                self.face = RIGHT[self.face]; log.append('Turn Right')
            self.actions += 1
        self.pos = c; self.actions += 1; log.append('Forward')


def run(world):
    a, out = Agent(), []
    visited_order = []
    while True:
        p = world.percept(a.pos)
        a.tell(a.pos, p)
        visited_order.append(a.pos)
        shown = [x if x else 'None' for x in p]
        out.append('Room %s  percept = [%s]' % (list(a.pos), ', '.join(shown)))
        if p[2]:
            a.has_gold = True; a.actions += 1
            out.append('   -> Glitter perceived: GRAB the gold')
            break
        wc = a.wumpus_cell()
        if wc: out.append(f'   -> Inference: Wumpus must be in {list(wc)}')
        pp = sorted(a.possible_pits())
        if pp: out.append(f'   -> Possible pits (P?): {[list(x) for x in pp]}')
        t = a.choose_target()
        if t is None:
            out.append('   -> No safe unvisited room left'); return out, a
        route = a.path(t)
        acts = []
        for c in route: a.step_to(c, acts)
        # walk through intermediate rooms silently, report the last
        out.append('   -> Safe move to %s  via %s  actions: %s' % (list(t), [list(r) for r in route], ', '.join(acts)))
        visited_order.extend(route[:-1])
        for c in route[:-1]:
            a.tell(c, world.percept(c))
    # return to start and climb
    route = a.path((1, 1)); acts = []
    for c in route: a.step_to(c, acts)
    a.actions += 1
    out.append('Return to [1, 1] via %s: %s, Climb' % ([list(r) for r in route], ', '.join(acts)))
    score = 1000 - a.actions
    out.append(f'Total actions = {a.actions}, Final score = 1000 - {a.actions} = {score}')
    return out, a


if __name__ == '__main__':
    w = World(wumpus=(1, 3), gold=(2, 3), pits=[(3, 1), (4, 4)])
    log, agent = run(w)
    print(*log, sep=chr(10))
