# CSCI 335 Project 2 Report

All times are average milliseconds per operation (ItemGenerator seed 42, `g++ -O2`).

## Part A: `contains()` (CompareItemName)

| n | vector | list | unordered_set | Tree |
|------|----------|----------|---------------|----------|
| 1000 | 0.000440 | 0.004169 | 0.000114 | 0.004085 |
| 2000 | 0.000975 | 0.008930 | 0.000112 | 0.010917 |
| 4000 | 0.002190 | 0.020878 | 0.000117 | 0.025797 |
| 8000 | 0.006409 | 0.042632 | 0.000176 | 0.061101 |

## Part B: `query()`

CompareItemName

| n | vector | list | unordered_set | Tree |
|------|----------|----------|---------------|----------|
| 1000 | 0.049000 | 0.052698 | 0.049780 | 0.039713 |
| 2000 | 0.092066 | 0.096410 | 0.100470 | 0.076130 |
| 4000 | 0.199794 | 0.207322 | 0.211888 | 0.167726 |
| 8000 | 0.577975 | 0.576980 | 0.696557 | 0.560025 |

CompareItemWeight

| n | vector | list | unordered_set | Tree |
|------|----------|----------|---------------|----------|
| 1000 | 0.006036 | 0.009205 | 0.009115 | 0.000696 |
| 2000 | 0.013645 | 0.017109 | 0.018507 | 0.001355 |
| 4000 | 0.038090 | 0.036127 | 0.043462 | 0.002257 |
| 8000 | 0.047724 | 0.097096 | 0.183043 | 0.004343 |

## Conclusion

For `contains()` I expected the hash set to win, and it did: its time stays essentially flat (about 0.0001 ms) as n grows, because hashing the name gives O(1) average lookup. The vector and list scale linearly with n, as expected from a linear scan, with the list roughly 5-7x slower than the vector because its nodes are scattered in memory and traversal has poor cache locality. The tree was the surprise: it was as slow as or slower than the list. The AVL is ordered by the Comparator, not by name, so `contains()` by name must visit every node (O(n)), and each visit is a pointer chase through the heap. This is the penalty described in the assignment, and it grows with n.

For `query()` the picture depends on the comparator. With CompareItemName, the random name ranges are wide and match a large fraction of the items, so every structure is dominated by the cost of copying matching items into the result set. Output size, not the search, drives the time, so all four are close (the tree is only slightly ahead; the hash set is a bit behind because iterating a hash table's buckets is slower). With CompareItemWeight the range is only 0.1 wide, so few items match. Here the tree is far faster, and the gap widens with n: vector, list and hash set must scan all n items (O(n)), while the AVL only visits the nodes along the boundary paths plus the matches (O(log n + k)). At n = 8000 the tree is roughly 11x faster than the vector and 40x faster than the hash set.

Use a hash-based inventory when the workload is dominated by exact lookups, insertions and removals by name: O(1) average and no ordering needed. Use a tree-based inventory when the workload involves repeated range queries on an ordered property (weight, name range), since a balanced tree skips whole subtrees and the cost scales with the result size and log n, and when you want items kept in sorted order. A hash set cannot exploit ordering at all, so range queries on it always degrade to a full scan, and a name lookup in this particular tree (ordered by something other than name) is no better than a list.
