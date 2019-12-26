using AdaGram

# println("Loading data")
vm, dict = load_model("/home/chaitanya/Research/ShataAnuvadak/_pytorch/all_indian_languages/data/monolingual/hindi/corpus.monolingual.hi.tok.model")
write_word2vec("/tmp/oo", vm, dict) #Dummy output file.
exit(0)

open("/tmp/word-context.txt") do file
    for ln in eachline(file)
        x = split(strip(ln))
        line_id = x[1]
        word_id = x[2]
        center_word = x[3]
        context = x[4:length(x)]
        try
            res = disambiguate(vm, dict, center_word, context)
            print(line_id, " ", word_id, " ")
            for j in 1:5
                print(res[j])
                print(" ")
            end
            println()
        catch y
            println(line_id, " ", word_id, " 1 0 0 0 0")
        end
    end
end
exit(0)
for i in 1:1206528
    print(dict.id2word[i])
    print(" ")
    a = expected_pi(vm, i)
    for j in 1:5
        print(a[j])
        print(" ")
    end
    println()
end
#save_model("./rawdata/o", vm, dict, 1e-5)
exit(0)
#println(vm.In[1, 1, 1])
#println(vm.In[2, 1, 1])
write_word2vec("./rawdata/wv.txt", vm, dict)
exit(0)
println(expected_pi(vm, dict.word2id["may"]))
println(nearest_neighbors(vm, dict, "may", 1, 5));
println(nearest_neighbors(vm, dict, "may", 2, 5));
println(nearest_neighbors(vm, dict, "may", 3, 5));
println(nearest_neighbors(vm, dict, "may", 4, 5));

write_word2vec("./rawdata/wv.txt", vm, dict)

a = expected_pi(vm, 1)
println(a)
println(a[1])
println(dict.id2word[1])
println(vm.In[:, 1, 1])
println()
println(vm.In[:, 2, 1])
println()
println(vm.In[:, 3, 1])
println()
println(vm.In[:, 4, 1])
println()
println(vm.In[:, 5, 1])
println()
