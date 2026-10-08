#include "ItemGenerator.hpp"

std::unordered_set<std::string> ItemGenerator::getUsedNames() const
{
    return used_names;
}

std::string ItemGenerator::randomChoice(const std::vector<std::string>& values)
{
    std::uniform_int_distribution<size_t> dist(0, values.size() - 1);
    return values[dist(rng)];
}

float ItemGenerator::randomFloat(const float& min, const float& max)
{
    std::uniform_real_distribution<float> dist(min, max);
    return std::round(dist(rng) * 10.0) / 10.0;
}

std::string ItemGenerator::randomUsedName()
{
    if (used_names.size() == 0) {
        return "";
    }
    std::uniform_int_distribution<int> dist(1, used_names.size());
    int randomIndex = dist(rng);

    for (auto itr = used_names.begin(); itr != used_names.end(); ++itr) {
        if ((--randomIndex) == 0) {
            return *itr;
        }
    }
    return "";
}

ItemType ItemGenerator::randomType()
{
    float roll = std::uniform_real_distribution<float>(0.0, 1.0)(rng);
    if (roll < 0.33)
        return WEAPON;
    if (roll < 0.66)
        return ACCESSORY;
    return ARMOR;
}

std::string ItemGenerator::generateUniqueName(ItemType type)
{
    const size_t ATTEMPTS = 3;
    for (size_t i = 0; i < ATTEMPTS; i++) {
        std::string name;

        name += randomChoice(prefixes) + " ";
        name += randomChoice(materials) + " ";

        const std::vector<std::string>* bank = nullptr;
        switch (type) {
        case WEAPON:
            bank = &weapon_types;
            break;
        case ARMOR:
            bank = &armor_types;
            break;
        case ACCESSORY:
            bank = &accessory_types;
            break;
        default:
            break;
        }

        if (bank) {
            name += randomChoice(*bank) + " ";
        }
        name += randomChoice(suffixes);

        if (used_names.count(name) == 0) {
            used_names.insert(name);
            return name;
        }
    }

    std::string generic = "Item_" + std::to_string(used_names.size());
    used_names.insert(generic);
    return generic;
}

ItemGenerator::ItemGenerator(std::uint32_t seed)
    : rng(seed)
{
}

Item ItemGenerator::randomItem()
{
    ItemType type = randomType();
    std::string name = generateUniqueName(type);
    float weight = randomFloat(MIN_WEIGHT, MAX_WEIGHT);
    return Item(name, weight, type);
}

std::unordered_set<Item> ItemGenerator::getRandomItems(size_t count)
{
    std::unordered_set<Item> generated;
    for (size_t i = 0; i < count; i++) {
        generated.insert(randomItem());
    }
    return generated;
}