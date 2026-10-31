#pragma once

#include "Item.hpp"
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <unordered_set>


class ItemGenerator {
private:
    std::mt19937 rng;
    std::unordered_set<std::string> used_names;

    // Word banks for name generation
    const std::vector<std::string> prefixes = {
        "Ancient", "Divine", "Celestial", "Infernal", "Shadow", "Frost", "Thunder",
        "Demonic", "Angelic", "Storm", "Astral", "Ethereal"
    };

    const std::vector<std::string> weapon_types = {
        "Sword", "Axe", "Mace", "Dagger", "Staff", "Bow", "Spear",
        "Glaive", "Scythe", "Warhammer", "Blade", "Greatsword"
    };

    const std::vector<std::string> armor_types = {
        "Helm", "Breastplate", "Gauntlets", "Boots", "Pauldrons",
        "Vambraces", "Cuirass", "Crown", "Shield", "Platemail"
    };

    const std::vector<std::string> accessory_types = {
        "Ring", "Amulet", "Necklace", "Bracelet", "Crown", "Circlet",
        "Pendant", "Charm", "Talisman", "Band", "Locket"
    };

    const std::vector<std::string> materials = {
        "Steel", "Iron", "Gold", "Silver", "Mithril", "Crystal",
        "Diamond", "Ruby", "Sapphire", "Emerald", "Obsidian"
    };

    const std::vector<std::string> suffixes = {
        "of Power", "of Might", "of the King", "of Light", "of Darkness",
        "of the Storm", "of the Earth", "of Time", "of the Ancients"
    };

    std::string randomChoice(const std::vector<std::string>& values);

    std::string generateUniqueName(ItemType type);

    std::unordered_set<std::string> getUsedNames() const;

public:
    static constexpr float MIN_WEIGHT = 0.1;
    static constexpr float MAX_WEIGHT = 30.0;
    float randomFloat(const float& min, const float& max);
    std::string randomUsedName();
    ItemType randomType();

    ItemGenerator(std::uint32_t seed);
    Item randomItem();
    std::unordered_set<Item> getRandomItems(size_t count);
};