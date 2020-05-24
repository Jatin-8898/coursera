package mastermind

data class Evaluation(val rightPosition: Int, val wrongPosition: Int)

fun evaluateGuess(secret: String, guess: String): Evaluation {

/*    print(secret.zip(guess))*/
    val positions = secret.zip(guess).count { c -> c.first == c.second }

    val commonLetters = "ABCDEF".sumBy {
        ch -> Math.min(secret.count { it == ch }, guess.count { it == ch }) }

    /*print(commonLetters)*/

    return Evaluation(positions, commonLetters - positions)
}

