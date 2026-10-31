#pragma once

#include <iostream>
#include <string>

enum ItemType {
    NONE = 0,
    WEAPON = 1,
    ACCESSORY = 2,
    ARMOR = 3
};

struct Item {
    std::string name_;
    float weight_;
    ItemType type_;

    /**
     * @brief Constructs a new Item.
     *
     * @param name The name of the item, default value of an empty string
     * @param weight The weight of the item, default value of 0
     * @param type The type of the item, defualt value ItemType::None
     */
    Item(const std::string& name = "", const float& weight = 0, const ItemType& type = NONE);

    /**
     * @brief Compares two Items for equality based on their names.
     *
     * @param rhs The Item to compare against
     * @return true if the items have the same name, false otherwise
     */
    bool operator==(const Item& rhs) const;

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
    friend std::ostream& operator<<(std::ostream& os, const Item& item);
};

/**
 * @brief Hash function for Item to support use in
 *        unordered containers. Implements hashing
 *        based on the item's name.
 *
 * @note Specialization of std::hash for Item (note the empty brackets for the template)
 */
template <>
struct std::hash<Item> {

    /**
     * @brief Computes a hash value for an Item based on
     * using the standard hash for strings on the Item's name
     *
     * @param i The Item to hash
     * @return Hash value for the Item
     */
    size_t operator()(const Item& i) const;
};