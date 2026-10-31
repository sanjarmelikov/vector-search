/**
 * @file ItemAVL.hpp
 * @brief Defines the interface for the AVLtree class & implementation of the Node struct
 */

#pragma once
#include <algorithm>
#include <iostream>
#include <iterator>
#include <vector>

#include "Compare.hpp"
#include "Item.hpp"
#include <queue>
#include <unordered_set>

struct Node {
    Item value_;
    int height_; // The height of the Node
    Node* left_; // A pointer to Node's left child
    Node* right_; // A pointer to Node's right child

    // Parameterized constructor for a Node
    Node(Item i, Node* lt = nullptr, Node* rt = nullptr)
        : value_ { i }
        , height_ { 0 }
        , left_ { lt }
        , right_ { rt }
    {
    }
};

template <class Comparator = CompareItemName>
class ItemAVL {
public:
    /**
     * @brief Default Constructor: Construct a new AVLtree object
     */
    ItemAVL();

    /**
     * @brief Destroy the AVLtree, deallocating all necessary Nodes
     */
    ~ItemAVL();

    // =========== INSERTION & BALANCE  ===========

    /**
     * @brief Gets the Node stored within the root
     * @return A const pointer to the Node representing the root of the ItemTree
     */
    const Node* root() const;

    /**
     * @brief Inserts an item to a tree, if there are no items
     * whose name equals the name of the item to insert
     *
     * @param target The Item to insert
     * @return True if the Item was successfully inserted, false otherwise.
     */
    bool insert(const Item& target);

    /**
     * @brief Erases the item whose name matches the target name
     *
     * @param name The name of the item to delete
     * @return The weight of the Item if it was successfully deleted, 0 otherwise.
     */
    float erase(const std::string& name);

    /**
     * @brief Checks if the tree contains an item with a given name
     * @param name The name to check for
     * @return True if a matching Item exists, false otherwise.
     */
    bool contains(const std::string& name) const;

    /**
     * @brief Determines the height of a given Node
     *
     * @param n A pointer to a node to be examined
     * @return The height of the given node, or -1 if given a nullptr
     */
    int height(Node* n) const;

    /**
     * @brief Prints level-order traversal of the tree
     */
    void displayLevelOrder() const;

    /**
     * @brief Prints the unique file sizes of the ItemAVL in-order
     */
    void displayInOrder() const;

    /**
     * @brief Returns the size of the AVL tree
     */
    int size() const;

private:
    static const int ALLOWED_IMBALANCE = 1;
    Node* root_;
    int size_;

    /**
     * @brief Internal routine to check if a subtree contains an item with a given name
     *
     * @param name The name to check for
     * @param subroot The root of the subtree to be searched
     * @return True if a matching Item exists, false otherwise.
     */
    bool contains(const std::string& name, const Node* subroot) const;

    /**
     * @brief Internal routine to insert into a subtree
     *
     * @param target The Item to insert
     * @param subroot The root of the subtree to be inserted into
     * @post Set the new root of the subtree
     */
    void insert(const Item& target, Node*& subroot);

    /**
     * @brief Internal routes for deletion in a subtree
     *
     * @param itemName The name of the item to delete
     * @param subroot The root of the subtree of which to search for the item to be deleted
     * @return The weight of the item if deleted. Otherwise, 0.
     * @post Set the new root of the subtree
     */
    float erase(const std::string& itemName, Node*& subroot);

    /**
     * @brief Balance the given Node
     *
     * @param t The Node to be balanced
     */
    void balance(Node*& t);

    /**
     * @brief Prints the level-order traversal of the specified subtree
     */
    void displayLevelOrder(Node* t) const;

    /**
     * @brief Helper for printInorder(). Prints out the contents of Items in-order.
     * @param root The root of the tree to be printed
     */
    void displayInOrder(Node* t) const;

    // =========== ROTATIONS  ===========

    /**
     * @brief Rotates a Node with its left child
     *
     * @param k2 The parent Node to be rotated
     * @post Updates heights and sets new root
     */
    void rotateWithLeftChild(Node*& k2);

    /**
     * @brief Rotates a Node with its right child
     *
     * @param k2 The parent Node to be rotated
     * @post Updates heights and sets new root
     */
    void rotateWithRightChild(Node*& k1);

    /**
     * @brief Performs a double rotation to fix a left-right imbalance about k3
     *
     * @param k3 The parent Node with a left-right imbalance
     * @post Updates heights, sets new root
     */
    void doubleWithLeftChlid(Node*& k3);

    /**
     * @brief Performs a double rotation to fix a right-left imbalance about k3
     *
     * @param k3 The parent Node with a right-left imbalance
     * @post Updates heights, sets new root
     */
    void doubleWithRightChild(Node*& k3);

    /**
     * @brief Destroys the given Node and its children
     *
     * @param t The node to be deleted
     */
    void deleteTree(Node*& t);
};

#include "ItemAVL.cpp"