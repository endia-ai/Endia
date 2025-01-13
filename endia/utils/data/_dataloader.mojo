from endia import Array, arange, contiguous, squeeze
from random import random_si64, seed


@value
struct _DataLoaderSpecs(CollectionElement):
    var storage: Array
    var curr_data: Array
    var shuffle: Bool
    var drop_last: Bool
    var num_rows: Int
    var num_cols: Int
    var batch_size: Int
    var indeces: List[Int]
    var index: Int
    var seed: Int

    fn __init__(
        inout self,
        curr_data: Array,
        shuffle: Bool,
        batch_size: Int,
        drop_last: Bool,
        seed: Int,
    ) raises:
        if curr_data.ndim() != 2:
            raise "Data must be 2D"

        self.storage = contiguous(curr_data)
        self.curr_data = Array(curr_data.shape())
        self.shuffle = shuffle
        self.num_rows = curr_data.shape()[0]
        self.num_cols = curr_data.shape()[1]
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.indeces = List[Int]()
        self.index = 0
        self.seed = seed

        for i in range(self.num_rows):
            self.indeces.append(i)

        if self.shuffle:
            self.shuffle_data()

    fn arange_indeces(inout self) raises:
        for i in range(self.num_rows):
            self.indeces[i] = i

    fn shuffle_indeces(inout self) raises:
        for i in range(self.num_rows):
            seed() if self.seed == 0 else seed(self.seed)
            var j = int(random_si64(i, self.num_rows - 1))
            var temp = self.indeces[i]
            self.indeces[i] = self.indeces[j]
            self.indeces[j] = temp

    fn shuffle_data(inout self) raises:
        self.shuffle_indeces() if self.shuffle else self.arange_indeces()

        var storage = self.storage.data()
        var curr_data = self.curr_data.data()

        var new_curr_data_len = self.num_rows if not self.drop_last else self.num_rows - self.num_rows % self.batch_size

        for i in range(new_curr_data_len):
            var src_row = i * self.num_cols
            var dest_row = self.indeces[i] * self.num_cols
            for k in range(self.num_cols):
                curr_data[src_row + k] = storage[dest_row + k]

    fn __getitem__(inout self, index: Int) raises -> Array:
        return contiguous(squeeze(self.curr_data[index : index + 1]))

    fn curr_data_(inout self, value: Array) raises:
        self.storage = contiguous(value)
        self.curr_data = Array(value.shape())
        self.num_rows = value.shape()[0]
        self.num_cols = value.shape()[1]
        self.index = 0
        self.shuffle_data()

    fn shuffle_(inout self, value: Bool) raises:
        self.shuffle = value
        self.shuffle_data()

    fn batch_size_(inout self, value: Int) raises:
        self.batch_size = value
        self.shuffle_data()

    fn drop_last_(inout self, value: Bool) raises:
        self.drop_last = value
        self.shuffle_data()

    fn seed_(inout self, value: Int) raises:
        self.seed = value
        self.shuffle_data()

    fn __len__(self) -> Int:
        return (
            self.num_rows
            - self.index if not self.drop_last else self.num_rows
            - self.index
            - self.num_rows % self.batch_size
        )


@value
struct _DatatLoaderIter(CollectionElement):
    var _data_loader_specs: _DataLoaderSpecs

    fn __init__(inout self, _data_loader_specs: _DataLoaderSpecs) raises:
        self._data_loader_specs = _data_loader_specs

    fn __len__(self) -> Int:
        return len(self._data_loader_specs)

    fn __iter__(inout self) raises -> Self:
        return self

    fn __next__(inout self) raises -> ArcPointer[Array]:
        var index = self._data_loader_specs.index
        var batch_size = self._data_loader_specs.batch_size

        if index + batch_size <= self._data_loader_specs.num_rows:
            var res = self._data_loader_specs.curr_data[
                index : index + batch_size
            ]
            self._data_loader_specs.index += batch_size
            return ArcPointer(contiguous(res))
        elif (
            index < self._data_loader_specs.num_rows
            and not self._data_loader_specs.drop_last
        ):
            var res = self._data_loader_specs.curr_data[
                index : self._data_loader_specs.num_rows
            ]
            self._data_loader_specs.index = self._data_loader_specs.num_rows
            return ArcPointer(contiguous(res))
        else:
            raise "StopIteration"


@value
struct DataLoader(CollectionElement):
    var _data_loader_specs: _DataLoaderSpecs
    var _data_loader_iter: _DatatLoaderIter

    fn __init__(
        inout self,
        curr_data: Array,
        shuffle: Bool = True,
        batch_size: Int = 1,
        drop_last: Bool = True,
        seed: Int = 0,
    ) raises:
        self._data_loader_specs = _DataLoaderSpecs(
            curr_data, shuffle, batch_size, drop_last, seed
        )
        self._data_loader_iter = _DatatLoaderIter(self._data_loader_specs)

    fn __iter__(inout self) raises -> _DatatLoaderIter:
        self._data_loader_specs.shuffle_data()
        return self._data_loader_iter

    fn __len__(self) -> Int:
        return len(self._data_loader_specs)

    fn __next__(inout self) raises -> Array:
        var index = self._data_loader_specs.index
        var batch_size = self._data_loader_specs.batch_size

        if index + batch_size <= self._data_loader_specs.num_rows:
            var res = self._data_loader_specs.curr_data[
                index : index + batch_size
            ]
            self._data_loader_specs.index += batch_size
            return contiguous(res)
        elif (
            index < self._data_loader_specs.num_rows
            and not self._data_loader_specs.drop_last
        ):
            var res = self._data_loader_specs.curr_data[
                index : self._data_loader_specs.num_rows
            ]
            self._data_loader_specs.index = self._data_loader_specs.num_rows
            return contiguous(res)
        else:
            print(" ")
            self._data_loader_specs.index = 0
            self._data_loader_specs.shuffle_data()
            return self.__next__()

    fn __getitem__(inout self, index: Int) raises -> Array:
        return contiguous(
            squeeze(self._data_loader_specs.curr_data[index : index + 1])
        )

    fn curr_data_(inout self, value: Array) raises:
        self._data_loader_specs.curr_data_(value)

    fn shuffle_(inout self, value: Bool) raises:
        self._data_loader_specs.shuffle_(value)

    fn drop_last_(inout self, value: Bool) raises:
        self._data_loader_specs.drop_last_(value)

    fn batch_size_(inout self, value: Int) raises:
        self._data_loader_specs.batch_size_(value)

    fn seed_(inout self, value: Int) raises:
        self._data_loader_specs.seed_(value)


fn next(inout data_loader: DataLoader) raises -> Array:
    return data_loader.__next__()
