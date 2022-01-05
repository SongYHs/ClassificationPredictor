import numpy as np

def generator_data(inputs, tags, batch_size, max_size, flag=False):
    size0 = inputs.shape[0]
    ii=0

    if not flag:
        while ii<max_size:
            # for j in range(int(10000000*np.random.randint(1,10))):
            #     pass
            # print(ii)
            start = ii % size0
            now = min([(ii+batch_size), max_size])  # 50000
            end =  now % size0 
            # print(start, end, now)

            if not int(ii / size0) == int(now / size0):
                data = np.append(inputs[start:], inputs[:end], axis=0) #[i*max_len:min([i*max_len+max_len, size])] 
                tag = np.append(tags[start:], tags[:end]) #[i*max_len:min([i*max_len+max_len, size])] 
            else:
                data = inputs[start:end]
                tag = tags[start:end]
            ii += batch_size
            yield data, tag
    else:
        while ii<max_size:
            start = ii
            end = min([ii+batch_size, max_size])
            ii += batch_size
            yield inputs[start: end], tags[start: end]

