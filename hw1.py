import random

class Portfolio(object):
    def __init__(self):
        self.cash = 0.00
        self.credit = {"stock": {}, "mutual funds": {}, "bonds": {}}
        self.hist = "Portfolio is set"

    def addCash(self, cash):
        self.cash += int(100*cash)/100.0
        self.hist += "Added $%.2f" % (int(100*cash)/100.0)

    def withdrawCash(self, cash):
        if cash > self.cash:
            print("Portfolio does not contain enough cash.")
        else:
            self.cash -= int(100*cash)/100.0
            self.hist += "Withdrew $%.2f" % (int(100*cash)/100.0)

    def buyAsset(self, number, asset):
        if self.cash < number*asset.price:
            print("Portfolio does not contain enough cash.")
            return None
        self.withdrawCash(number*asset.price)
        if asset in self.credit[asset.getClass()]:
            self.credit[asset.getClass()][asset] += number
        else:
            self.credit[asset.getClass()][asset] = number
        self.hist += "Bought %d of %s named %s" % (
            number, asset.getClass(), asset.name)


    def buyStock(self, number, asset): self.buyAsset(int(number), asset)

    buyMutualFund = buyBonds = buyAsset

    def sellAsset(self, number, asset):
        if asset in self.credit[asset.getClass()]:
            if self.credit[asset.getClass()][asset] < number:
                print("The portfolio does not contain enough of %s %s" %
                      (asset.name, asset.getClass()))
            else:
                self.credit[asset.getClass()][asset] -= number
                if self.credit[asset.getClass()][asset] == 0:
                    del self.credit[asset.getClass()][asset]
                self.addCash(number*asset.SellPrice())
                self.hist += "Sold %d of %s named %s" % (
                    number, asset.getClass(), asset.name)
        else:
            print("The portfolio does not contain %s with name %s" %
                  (asset.getClass(), asset.name))

    def sellStock(self, number, asset): self.sellAsset(
        int(number), asset)

    sellMutualFund = sellBonds = sellAsset

    def __str__(self):
        output = "cash: $%-15.2f" % self.cash
        for asset in self.credit:
            output += "%s: " % asset
            if not self.credit[asset]:
                output += '\tnone'
            for ast in self.credit[asset]:
                output += str(self.credit[asset][ast]
                              ).rjust(5) + str(ast.name).rjust(5) + ""
        return output

    def history(self): print(self.hist)


class Asset(object):
    def __init__(self, price, name):
        self.price = price
        self.name = name

    def SellPrice(self):
        return int(100*random.uniform(.9*self.price, 1.2*self.price))/100.0


class Stock(Asset):
    def __init__(self, price, name):
        Asset.__init__(self, price, name)


    def getClass(self): return "stock"

    def SellPrice(self):
        return int(100*random.uniform(.5*self.price, 1.5*self.price))/100.0


class MutualFund(Asset):
    def __init__(self, name):
        Asset.__init__(self, 1.0, name)

    def getClass(self): return "mutual funds"


class Bonds(Asset):
    def __init__(self, price, name):
        Asset.__init__(self, price, name)

    def getClass(self): return "bonds"
