#include "Compare.hpp"
#include "HashInventory.hpp"
#include "Inventory.hpp"
#include "Item.hpp"
#include "ItemAVL.hpp"
#include "ItemGenerator.hpp"
#include "TreeInventory.hpp"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <list>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;
volatile size_t sink = 0;

template <class Func>
double timeMs(Func&& f)
{
    auto t0 = Clock::now();
    f();
    auto t1 = Clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

template <class Inv>
double timeContains(const Inv& inv, const std::vector<std::string>& names)
{
    double total = 0;
    for (const auto& name : names) {
        bool found = false;
        total += timeMs([&] { found = inv.contains(name); });
        sink = sink + found;
    }
    return total;
}

template <class Inv>
double timeQuery(const Inv& inv, const Item& start, const Item& end)
{
    std::unordered_set<Item> result;
    double t = timeMs([&] { result = inv.query(start, end); });
    sink = sink + result.size();
    return t;
}

template <class Inv>
double avgContains(size_t n)
{
    Inv inv;
    ItemGenerator gen(42);
    for (size_t i = 0; i < n; ++i) {
        inv.pickup(gen.randomItem());
    }
    std::vector<std::string> contained, missing;
    for (int i = 0; i < 100; ++i) {
        contained.push_back(gen.randomUsedName());
    }
    for (int i = 0; i < 100; ++i) {
        missing.push_back(gen.randomItem().name_);
    }
    double total = timeContains(inv, contained) + timeContains(inv, missing);
    return total / 200.0;
}

template <class Inv>
double avgQueryName(size_t n)
{
    Inv inv;
    ItemGenerator gen(42);
    for (size_t i = 0; i < n; ++i) {
        inv.pickup(gen.randomItem());
    }
    double total = 0;
    for (int i = 0; i < 10; ++i) {
        Item start(gen.randomUsedName());
        Item end(gen.randomUsedName());
        if (end.name_ < start.name_) {
            std::swap(start, end);
        }
        total += timeQuery(inv, start, end);
    }
    return total / 10.0;
}

template <class Inv>
double avgQueryWeight(size_t n)
{
    Inv inv;
    ItemGenerator gen(42);
    for (size_t i = 0; i < n; ++i) {
        inv.pickup(gen.randomItem());
    }
    double total = 0;
    for (int i = 0; i < 10; ++i) {
        float w = gen.randomFloat(ItemGenerator::MIN_WEIGHT, ItemGenerator::MAX_WEIGHT);
        Item start("", w, NONE);
        Item end("", w + 0.1f, NONE);
        total += timeQuery(inv, start, end);
    }
    return total / 10.0;
}


template <class Comparator>
using VecInv = Inventory<Comparator>;
template <class Comparator>
using ListInv = Inventory<Comparator, std::list<Item>>;
template <class Comparator>
using HashInv = Inventory<Comparator, std::unordered_set<Item>>;
template <class Comparator>
using TreeInv = Inventory<Comparator, Tree>;

int main()
{
    const std::vector<size_t> ns = { 1000, 2000, 4000, 8000 };
    std::cout << std::fixed << std::setprecision(6);

    std::cout << "Part A: avg contains() time (ms), CompareItemName\n";
    std::cout << "n\tvector\tlist\tunordered_set\ttree\n";
    for (size_t n : ns) {
        std::cout << n << '\t'
                  << avgContains<VecInv<CompareItemName>>(n) << '\t'
                  << avgContains<ListInv<CompareItemName>>(n) << '\t'
                  << avgContains<HashInv<CompareItemName>>(n) << '\t'
                  << avgContains<TreeInv<CompareItemName>>(n) << '\n';
    }

    std::cout << "\nPart B: avg query() time (ms), CompareItemName\n";
    std::cout << "n\tvector\tlist\tunordered_set\ttree\n";
    for (size_t n : ns) {
        std::cout << n << '\t'
                  << avgQueryName<VecInv<CompareItemName>>(n) << '\t'
                  << avgQueryName<ListInv<CompareItemName>>(n) << '\t'
                  << avgQueryName<HashInv<CompareItemName>>(n) << '\t'
                  << avgQueryName<TreeInv<CompareItemName>>(n) << '\n';
    }

    std::cout << "\nPart B: avg query() time (ms), CompareItemWeight\n";
    std::cout << "n\tvector\tlist\tunordered_set\ttree\n";
    for (size_t n : ns) {
        std::cout << n << '\t'
                  << avgQueryWeight<VecInv<CompareItemWeight>>(n) << '\t'
                  << avgQueryWeight<ListInv<CompareItemWeight>>(n) << '\t'
                  << avgQueryWeight<HashInv<CompareItemWeight>>(n) << '\t'
                  << avgQueryWeight<TreeInv<CompareItemWeight>>(n) << '\n';
    }
    return 0;
}
