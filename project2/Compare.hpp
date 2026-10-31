#pragma once
#include "Item.hpp"

/**
 * @class CompareItemName
 * @brief Provides static comparison functions for Item names
 */
class CompareItemName {
public:
    /**
     * @brief Compares two items by name
     * @param a First item to compare
     * @param b Second item to compare
     * @return true if a's name is less than b's name
     */
    static bool lessThan(const Item& a, const Item& b);

    /**
     * @brief Checks if two items have equal names
     * @param a First item to compare
     * @param b Second item to compare
     * @return true if items have equal names
     */
    static bool equal(const Item& a, const Item& b);

    /**
     * @brief Checks if first item's name is less than or equal to second item's
     * name
     * @param a First item to compare
     * @param b Second item to compare
     * @return true if a's name is less than or equal to b's name
     */
    static bool leq(const Item& a, const Item& b);
};

/**
 * @class CompareItemWeight
 * @brief Provides static comparison functions for Item weights
 */
class CompareItemWeight {
public:
    /**
     * @brief Compares two items by weight
     * @param a First item to compare
     * @param b Second item to compare
     * @return true if a's weight is less than b's weight
     */
    static bool lessThan(const Item& a, const Item& b);

    /**
     * @brief Checks if two items have equal weights
     * @param a First item to compare
     * @param b Second item to compare
     * @return true if items have equal weights within epsilon
     */
    static bool equal(const Item& a, const Item& b);

    /**
     * @brief Checks if first item's weight is less than or equal to second item's
     * weight
     * @param a First item to compare
     * @param b Second item to compare
     * @return true if a's weight is less than or equal to b's weight
     */
    static bool leq(const Item& a, const Item& b);
};

/**
 * @class CompareItemType
 * @brief Provides static comparison functions for Item types
 */
class CompareItemType {
public:
    /**
     * @brief Compares two items by type
     * @param a First item to compare
     * @param b Second item to compare
     * @return true if a's type is less than b's type
     */
    static bool lessThan(const Item& a, const Item& b);

    /**
     * @brief Checks if two items have equal types
     * @param a First item to compare
     * @param b Second item to compare
     * @return true if items have equal types
     */
    static bool equal(const Item& a, const Item& b);

    /**
     * @brief Checks if first item's type is less than or equal to second item's
     * type
     * @param a First item to compare
     * @param b Second item to compare
     * @return true if a's type is less than or equal to b's type
     */
    static bool leq(const Item& a, const Item& b);
};