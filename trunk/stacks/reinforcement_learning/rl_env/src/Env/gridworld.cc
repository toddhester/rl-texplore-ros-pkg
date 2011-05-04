#include <rl_env/gridworld.hh>
#include <cmath>

std::ostream &operator<<(std::ostream &out, const Gridworld &g) {
  for (unsigned i = 0; i < g.ns.size(); ++i)
    out << " -";
  out << " \n";

  for (unsigned h = g.ew.size() - 1; h > 0; --h) {
    out << "| ";
    for (unsigned w = 0; w < g.ew[h].size(); ++w)
      out << (g.ew[h][w] ? "| " : "  ");
    out << "|\n";

    for (unsigned w = 0; w < g.ns.size(); ++w)
      out << (g.ns[w][h-1] ? " -" : "  ");
    out << " \n";
  }

  out << "| ";
  for (unsigned w = 0; w < g.ew[0].size(); ++w)
    out << (g.ew[0][w] ? "| " : "  ");
  out << "|\n";

  for (unsigned i = 0; i < g.ns.size(); ++i)
    out << " -";
  out << " \n";

  return out;
}

Gridworld::Gridworld(unsigned height, unsigned width, 
		     const std::vector<std::vector<bool> > &northsouth,
		     const std::vector<std::vector<bool> > &eastwest):
  h(height), w(width), ns(northsouth), ew(eastwest)
{}

Gridworld::Gridworld(unsigned height, unsigned width, Random &rng):
  h(height), w(width),
  ns(w, std::vector<bool>(h - 1, false)),
  ew(h, std::vector<bool>(w - 1, false))
{
  const unsigned n = static_cast<unsigned>(sqrt(static_cast<float>(w*h)));
  for (unsigned i = 0; i < n; ++i)
    add_obstacle(rng);
}

bool Gridworld::wall(unsigned nsCoord, unsigned ewCoord, unsigned dir) const {
  const bool isNS = 0 == dir/2;
  const bool isIncr = 0 == dir%2;
  const std::vector<std::vector<bool> > &walls = isNS ? ns : ew;
  unsigned major = isNS ? ewCoord : nsCoord;
  unsigned minor = isNS ? nsCoord : ewCoord;
  if (!isIncr) {
    if (minor == 0)
      return true;
    --minor;
  }
  if (minor >= walls[major].size())
    return true;
  return walls[major][minor];
}

void Gridworld::add_obstacle(Random &rng) {
  bool direction = rng.bernoulli(0.5);
  std::vector<std::vector<bool> > &parallel = direction ? ns : ew;
  std::vector<std::vector<bool> > &perpendicular = direction ? ew : ns;

  unsigned seedi = rng.uniformDiscrete(1, parallel.size()) - 1;
  unsigned seedj = rng.uniformDiscrete(1, parallel[seedi].size()) - 1;

  unsigned first = seedi + 1;
  while (isClear(first - 1, seedj, parallel, perpendicular))
    --first;
  unsigned last = seedi;
  while (isClear(last + 1, seedj, parallel, perpendicular))
    ++last;

  chooseSegment(first, last, seedj, parallel, rng);
}

void Gridworld::chooseSegment(unsigned first,
			      unsigned last,
			      unsigned j,
			      std::vector<std::vector<bool> > &parallel,
			      Random &rng)
{
  if (last <= first)
    return;
  unsigned maxLength = last - first;
  if (maxLength >= parallel.size())
    maxLength = parallel.size() - 1;
  unsigned length = maxLength > 1 ? rng.uniformDiscrete(1,maxLength) : 1;
  int dir = 1 - 2*rng.uniformDiscrete(0,1);
  unsigned start = (dir > 0) ? first : (last - 1);

  for (unsigned i = 0; i < length; ++i)
    parallel[start + i*dir][j] = true;
}
