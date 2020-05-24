package nicestring;

fun String.isNice2(): Boolean {

    fun firstCondition(): Boolean {
        //If !contains substr then rtn true
        if (!this.contains("bu") &&
                !this.contains("ba") &&
                !this.contains("be")) return true
        return false
    }

    fun secondCondition(): Boolean {
        // Check the count of vowel >= 3
        if (this.filter { c -> c == 'a' || c == 'e' || c == 'i' ||
                        c  == 'o' || c == 'u' }
                        .count() >= 3) return true
        return false
    }

    fun thirdCondition(): Boolean {
        // IF both the iterators eg a , b are same its a Double
        this.zipWithNext { a, b -> if (a == b) return true }
        return false
    }

    // If any of the 2 conditions satisfy return true
    if (firstCondition() && secondCondition() ||
            firstCondition() && thirdCondition() ||
            secondCondition() && thirdCondition()) return true
    return false

}