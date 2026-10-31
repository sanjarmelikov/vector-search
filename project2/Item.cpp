#include "Item.hpp"

/**
 * @brief Constructs a new Item.
 *
 * @param name The name of the item, default value of an empty string
 * @param weight The weight of the item, default value of 0
 * @param type The type of the item, defualt value ItemType::None
 */
Item::Item(const std::string& name, const float& weight, const ItemType& type)
    : name_ { name }
    , weight_ { weight }
    , type_ { type }
{
}

/**
 * @brief Compares two Items for equality based on their names.
 *
 * @param rhs The Item to compare against
 * @return true if the items have the same name, false otherwise
 */
bool Item::operator==(const Item& rhs) const
{
    return name_ == rhs.name_;
}

/**
 * @brief Outputs the Item's name to an output stream.
 *
 * @param os The output stream
 * @param item The Item to output
 * @return Reference to the output stream
 * @example
 * This allows us to pass the object directly to std::cout
 * So for example, we can call:
 *
 * Item a("MyItem", 1.0, ItemType::Weapon);
 * std::cout << a << std::endl;
 *
 * Which would output: MyItem
 */
std::ostream& operator<<(std::ostream& os, const Item& item)
{
    os << item.name_;
    return os;
}

/**
 * @brief Computes a hash value for an Item based on
 * using the standard hash for strings on the Item's name
 *
 * @param i The Item to hash
 * @return Hash value for the Item
 */
size_t std::hash<Item>::operator()(const Item& i) const
{
    return std::hash<std::string> {}(i.name_);
}