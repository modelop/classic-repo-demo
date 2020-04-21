# fastscore.schema.0: double
# fastscore.schema.1: double

# modelop.score
action <- function(datum){
    emit(datum + 1)
}
