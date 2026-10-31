#include "Inventory.hpp"

/**
 * @brief Default constructor for the Inventory template class.
 *
 * Initializes an empty inventory with no items, no equipped item,
 * and zero total weight.
 *
 * @tparam Comparator The comparison class for querying items
 * @tparam Container The container type used to store inventory items
 */
template <class Comparator, class Container>
Inventory<Comparator, Container>::Inventory()
    : items_ { Container() }
    , equipped_ { nullptr }
    , weight_ { 0.0 }
{
}

/**
 * @brief Retrieves the value stored in `equipped_`
 * @return The Item pointer stored in `equipped_`
 */
template <class Comparator, class Container>
Item* Inventory<Comparator, Container>::getEquipped() const
{
    return equipped_;
}

/**
 * @brief Equips a new item.
 * @param itemToEquip A pointer to the item to equip.
 * @post Updates `equipped` to the specified item
 * without deallocating the original.
 */
template <class Comparator, class Container>
void Inventory<Comparator, Container>::equip(Item* itemToEquip)
{
    equipped_ = itemToEquip;
}

/**
 * @brief Discards the currently equipped item.
 * @post Deallocates the item pointed to by `equipped`
 * and sets `equipped` to nullptr, if `equipped` is not nullptr already.
 */
template <class Comparator, class Container>
void Inventory<Comparator, Container>::discardEquipped()
{
    if (!equipped_) {
        return;
    }
    delete equipped_;
    equipped_ = nullptr;
}

/**
 * @brief Retrieves the value stored in `weight_`
 * @return The float value stored in `weight_`
 */
template <class Comparator, class Container>
float Inventory<Comparator, Container>::getWeight() const
{
    return weight_;
}

/**
 * @brief Retrieves the value stored in `item_count_`
 * @return The size_t value stored in `item_count_`
 */
template <class Comparator, class Container>
size_t
Inventory<Comparator, Container>::size() const
{
    return items_.size();
}

/**
 * @brief Retrieves a copy of the container holding inventory items.
 *
 * @return Container of items in the inventory
 */
template <class Comparator, class Container>
Container
Inventory<Comparator, Container>::getItems() const
{
    return items_;
}

/**
 * @brief Attempts to add a new item to the inventory.
 *
 * @param target Item to be added to the inventory
 * @return true if the item was successfully added, false if an item
 *         with the same name already exists
 */
template <class Comparator, class Container>
bool Inventory<Comparator, Container>::pickup(const Item& target)
{
    if (contains(target.name_)) {
        return false;
    }
    items_.insert(items_.end(), target);
    weight_ += target.weight_;
    return true;
}

/**
 * @brief Checks if an item with the given name exists in the inventory.
 *
 * @param name Name of the item to search for
 * @return true if the item exists in the inventory, false otherwise
 */
template <class Comparator, class Container>
bool Inventory<Comparator, Container>::contains(const std::string& name) const
{
    auto itr = std::find_if(items_.begin(), items_.end(), [&name](const Item& i) {
        return i.name_ == name;
    });

    if (itr == items_.end()) {
        return false;
    }
    return true;
}

/**
 * @brief Attempts to remove an item from the inventory by name.
 *
 * @param name Name of the item to be removed
 * @return true if the item was successfully removed, false if the
 *         item was not found in the inventory
 */
template <class Comparator, class Container>
bool Inventory<Comparator, Container>::discard(const std::string& name)
{
    auto itr = std::find_if(items_.begin(), items_.end(), [&name](const Item& i) {
        return i.name_ == name;
    });

    if (itr == items_.end()) {
        return false;
    }
    weight_ -= itr->weight_;
    items_.erase(itr);
    return true;
}

/**
 * @brief Queries the inventory for items within a specified range.
 *
 * Returns a set of items that fall between the start and end items
 * according to the specified Comparator (inclusive on both ends)
 *
 * @param start An Item whose compared property is the lower bound of the query range
 * @param end An Item whose compared property is the upper bound of the query range
 * @return std::unordered_set of items within the specified range
 *
 * @note Returns an empty set if the end item is less than the start item
 * @example To select all Items with weights 0.4 to 10.9, we'd setup the class & parameters as such:
 *  - this Inventory object is of type Inventory<CompareItemWeight>
 *  - start = Item("some_name", 0.4, ItemType::None)
 *  - end = = Item("some_other_name", 10.9, ItemType::None)
 *
 */
template <class Comparator, class Container>
std::unordered_set<Item>
Inventory<Comparator, Container>::query(const Item& start,
    const Item& end) const
{
    if (Comparator::lessThan(end, start)) {
        return {};
    }
    std::unordered_set<Item> matching;

    std::for_each(items_.begin(),
        items_.end(),
        [&start, &end, &matching, this](const Item& i) {
            bool isMatching = Comparator::leq(start, i) && Comparator::leq(i, end);
            if (isMatching) {
                matching.insert(i);
            }
        });

    return matching;
}

/**
 * @brief Destructor for the Inventory class.
 * @post Deallocates any dynamically allocated resources.
 */
template <class Comparator, class Container>
Inventory<Comparator, Container>::~Inventory()
{
    discardEquipped();
}