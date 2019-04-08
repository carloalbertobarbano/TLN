#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>

int main(int argc, char **argv) {
  std::ifstream input("paolofrancesca.cfg");

  std::vector<std::string> lines;
  for(std::string line; getline(input, line);)
    lines.push_back(line);

  lines.erase(
    std::remove_if(
      lines.begin(), lines.end(), 
      [](std::string line){return line[0] == '#' || line.empty();}
    ), lines.end()
  );

  for (auto line : lines)
    std::cout << line << std::endl;

  return 0;
}