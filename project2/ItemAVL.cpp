#include "ItemAVL.hpp"

template <class Comparator>
const Node* ItemAVL<Comparator>::root() const
{
    return root_;
}

/**
 * @brief Determines the height of a given Node
 *
 * @param n A pointer to a node to be examined
 * @return The height of the given node, or -1 if given a nullptr
 */
template <class Comparator>
int ItemAVL<Comparator>::height(Node* n) const
{
    if (n == nullptr) {
        return -1;
    }
    return n->height_;
}

/**
 * @brief Returns the size of the AVL tree
 */
template <class Comparator>
int ItemAVL<Comparator>::size() const
{
    return size_;
}

/**
 * @brief Prints the value of the specified Node t and its children using level-order traversal
 */
template <class Comparator>
void ItemAVL<Comparator>::displayLevelOrder(Node* t) const
{
    if (!t) {
        return;
    }

    std::queue<Node*> current, next;
    current.push(t);

    while (!current.empty()) {
        while (!current.empty()) {
            Node*& front = current.front();
            if (front->left_) {
                next.push(front->left_);
            }
            if (front->right_) {
                next.push(front->right_);
            }
            std::cout << front->value_ << std::endl;
            current.pop();
        }
        std::cout << std::endl;
        current = std::move(next);
    }
}

/**
 * @brief Prints the level-order traversal of the specified subtree
 */
template <class Comparator>
void ItemAVL<Comparator>::displayLevelOrder() const
{
    displayLevelOrder(root_);
}

/**
 * @brief Wrapper for printing the unique file sizes of the ItemAVL in-order
 */
template <class Comparator>
void ItemAVL<Comparator>::displayInOrder() const
{
    if (!root_) {
        return;
    }
    displayInOrder(root_);
    std::cout << std::endl;
}

/**
 * @brief Helper for displayInOrder(). Prints the unique file sizes in-order given a specified root
 * @param root The root of the tree to be printed
 */
template <class Comparator>
void ItemAVL<Comparator>::displayInOrder(Node* root) const
{
    if (!root) {
        return;
    }

    displayInOrder(root->left_);
    std::cout << root->value_ << std::endl;
    displayInOrder(root->right_);
}

/**
 * @brief Default Constructor: Construct a new ItemAVL object
 */
template <class Comparator>
ItemAVL<Comparator>::ItemAVL()
    : root_ { nullptr }
    , size_ { 0 }
{
}

/**
 * @brief Destroys the given Node and its children
 *
 * @param t The node to be deleted
 */
template <class Comparator>
void ItemAVL<Comparator>::deleteTree(Node*& t)
{
    if (t == nullptr) {
        return;
    }
    deleteTree(t->left_);
    deleteTree(t->right_);
    delete t;
    t = nullptr;
}

/**
 * @brief Destroy the ItemAVL, deallocating all necessary Nodes
 */
template <class Comparator>
ItemAVL<Comparator>::~ItemAVL()
{
    deleteTree(root_);
}

/**
 * @brief Checks if the tree contains an item with a given name
 * @param name The name to check for
 * @return True if a matching Item exists, false otherwise.
 */
template <class Comparator>
bool ItemAVL<Comparator>::contains(const std::string& target) const
{
    // Your code here
}

/**
 * @brief Internal routine to check if a subtree contains an item with a given name
 *
 * @param name The name to check for
 * @param subroot The root of the subtree to be searched
 * @return True if a matching Item exists, false otherwise.
 */
template <class Comparator>
bool ItemAVL<Comparator>::contains(const std::string& target, const Node* subroot) const
{
    // Your code here.
}

/**
 * @brief Inserts an item to a tree, if there are no items
 * whose name equals the name of the item to insert
 *
 * @param target The Item to insert
 * @return True if the Item was successfully inserted, false otherwise.
 */
template <class Comparator>
bool ItemAVL<Comparator>::insert(const Item& target)
{
    if (contains(target.name_)) {
        return false;
    }
    insert(target, root_);
    size_++;
    return true;
}

/**
 * @brief Internal routine to insert into a subtree
 *
 * @param target The value to insert
 * @param subroot The root of the subtree to be inserted into
 * @post Set the new root of the subtree
 */
template <class Comparator>
void ItemAVL<Comparator>::insert(const Item& target, Node*& subroot)
{
    if (subroot == nullptr) {
        subroot = new Node({ target });
        return;
    }

    if (Comparator::lessThan(target, subroot->value_)) {
        insert(target, subroot->left_);
    } else {
        insert(target, subroot->right_);
    }

    balance(subroot);
}

/**
 * @brief Erases the item whose name matches the target name
 *
 * @param name The name of the item to delete
 * @return The weight of the Item if it was successfully deleted, 0 otherwise.
 */
template <class Comparator>
float ItemAVL<Comparator>::erase(const std::string& target)
{
    float erased_weight = erase(target, root_);

    if (erased_weight) {
        size_--;
    }
    return erased_weight;
}

/**
 * @brief Internal routes for deletion in a subtree
 *
 * @param itemName The name of the item to delete
 * @param subroot The root of the subtree of which to search for the item to be deleted
 * @return The weight of the item if deleted. Otherwise, 0.
 * @post Set the new root of the subtree
 */
template <class Comparator>
float ItemAVL<Comparator>::erase(const std::string& target, Node*& subroot)
{
    if (subroot == nullptr) {
        return 0;
    }

    if (subroot->value_.name_ != target) {
        float erased_weight = erase(target, subroot->left_);
        if (erased_weight) {
            return erased_weight;
        }
        erased_weight = erase(target, subroot->right_);
        return erased_weight;
    }

    auto findMin = [](Node* subroot) {
        while (subroot && subroot->left_) {
            subroot = subroot->left_;
        }
        return subroot;
    };

    if (subroot->left_ && subroot->right_) {
        // Two children
        std::swap(subroot->value_, findMin(subroot->right_)->value_);
        return erase(target, subroot->right_);
    }
    Node* toDelete = subroot;
    float erased_weight = subroot->value_.weight_;
    subroot = (subroot->left_) ? subroot->left_ : subroot->right_;
    delete toDelete;
    toDelete = nullptr;

    balance(subroot);
    return erased_weight;
}

/**
 * @brief Balance the given Node
 *
 * @param t The Node to be balanced
 */
template <class Comparator>
void ItemAVL<Comparator>::balance(Node*& t)
{
    if (t == nullptr) {
        return;
    }

    if (height(t->left_) - height(t->right_) > ALLOWED_IMBALANCE) {
        if (height(t->left_->left_) >= height(t->left_->right_)) {
            rotateWithLeftChild(t);
        } else {
            doubleWithLeftChlid(t);
        }
    } else {
        if (height(t->right_) - height(t->left_) > ALLOWED_IMBALANCE) {
            if (height(t->right_->right_) >= height(t->right_->left_))
                rotateWithRightChild(t);
            else {
                doubleWithRightChild(t);
            }
        }
    }

    t->height_ = std::max(height(t->left_), height(t->right_)) + 1;
}

/**
 * @brief Performs a rotation on k2 with its left child
 *
 * @param k2 The node to be brought down to its left child
 * @post k2 is set to the rotated root (ie. its initial left child);
 * Both nodes roots are updated to reflect the rotation
 */
template <class Comparator>
void ItemAVL<Comparator>::rotateWithLeftChild(Node*& k2)
{
    Node* k1 = k2->left_;
    k2->left_ = k1->right_;
    k1->right_ = k2;

    k2->height_ = std::max(height(k2->left_), height(k2->right_)) + 1;
    k1->height_ = std::max(height(k1->left_), k2->height_) + 1;
    k2 = k1;
}

/**
 * @brief Performs a rotation on k1 with its right child
 *
 * @param k1 The node to be brought down to its right child
 * @post k1 is set to the rotated root (ie. its initial right child);
 * Both nodes roots are updated to reflect the rotation
 */
template <class Comparator>
void ItemAVL<Comparator>::rotateWithRightChild(Node*& k1)
{
    Node* k2 = k1->right_;
    k1->right_ = k2->left_;
    k2->left_ = k1;
    k1->height_ = 1 + std::max(height(k1->left_), height(k1->right_));
    k2->height_ = 1 + std::max(k1->height_, height(k2->right_));
    k1 = k2;
}

/**
 * @brief Performs a double rotation to fix a left-right imbalance about k3
 *
 * @param k3 The parent Node with a left-right imbalance
 * @post Updates heights, sets new root
 */
template <class Comparator>
void ItemAVL<Comparator>::doubleWithLeftChlid(Node*& k3)
{
    rotateWithRightChild(k3->left_);
    rotateWithLeftChild(k3);
}

/**
 * @brief Performs a double rotation to fix a right-left imbalance about k3
 *
 * @param k3 The parent Node with a right-left imbalance
 * @post Updates heights, sets new root
 */
template <class Comparator>
void ItemAVL<Comparator>::doubleWithRightChild(Node*& k3)
{
    rotateWithLeftChild(k3->right_);
    rotateWithRightChild(k3);
}